# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
from mmcv.cnn import build_conv_layer
from mmengine.dist import get_dist_info
from mmengine.structures import PixelData
from torch import Tensor, nn

from mmpose.codecs.utils import get_simcc_normalized
from mmpose.evaluation.functional import simcc_pck_accuracy
from mmpose.models.utils.tta import flip_vectors
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import ConfigType, InstanceList, OptConfigType, OptSampleList

# from ..base_head import BaseHead
from mmpose.models.heads.base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]


class SimCCHead(BaseHead):
    """Top-down heatmap head introduced in `SimCC`_ by Li et al (2022). The
    head is composed of a few deconvolutional layers followed by a fully-
    connected layer to generate 1d representation from low-resolution feature
    maps.

    Args:
        in_channels (int | sequence[int]): Number of channels in the input
            feature map
        out_channels (int): Number of channels in the output heatmap
        input_size (tuple): Input image size in shape [w, h]
        in_featuremap_size (int | sequence[int]): Size of input feature map
        simcc_split_ratio (float): Split ratio of pixels
        deconv_type (str, optional): The type of deconv head which should
            be one of the following options:

                - ``'heatmap'``: make deconv layers in `HeatmapHead`
                - ``'vipnas'``: make deconv layers in `ViPNASHead`

            Defaults to ``'Heatmap'``
        deconv_out_channels (sequence[int]): The output channel number of each
            deconv layer. Defaults to ``(256, 256, 256)``
        deconv_kernel_sizes (sequence[int | tuple], optional): The kernel size
            of each deconv layer. Each element should be either an integer for
            both height and width dimensions, or a tuple of two integers for
            the height and the width dimension respectively.Defaults to
            ``(4, 4, 4)``
        deconv_num_groups (Sequence[int], optional): The group number of each
            deconv layer. Defaults to ``(16, 16, 16)``
        conv_out_channels (sequence[int], optional): The output channel number
            of each intermediate conv layer. ``None`` means no intermediate
            conv layer between deconv layers and the final conv layer.
            Defaults to ``None``
        conv_kernel_sizes (sequence[int | tuple], optional): The kernel size
            of each intermediate conv layer. Defaults to ``None``
        final_layer (dict): Arguments of the final Conv2d layer.
            Defaults to ``dict(kernel_size=1)``
        loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KLDiscretLoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings

    .. _`SimCC`: https://arxiv.org/abs/2107.03332
    """

    _version = 2

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        input_size: Tuple[int, int],
        in_featuremap_size: Tuple[int, int],
        # simcc_split_ratio: float = 2.0,
        simcc_x_samples: int,
        simcc_y_samples: int,
        deconv_type: str = "heatmap",
        deconv_out_channels: OptIntSeq = (256, 256, 256),
        deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
        deconv_num_groups: OptIntSeq = (16, 16, 16),
        conv_out_channels: OptIntSeq = None,
        conv_kernel_sizes: OptIntSeq = None,
        final_layer: dict = dict(kernel_size=1),
        loss: ConfigType = dict(type="KLDiscretLoss", use_target_weight=True),
        # decoder: OptConfigType = None,
        init_cfg: OptConfigType = None,
    ):
        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        if deconv_type not in {"heatmap", "vipnas"}:
            raise ValueError(
                f"{self.__class__.__name__} got invalid `deconv_type` value"
                f"{deconv_type}. Should be one of "
                '{"heatmap", "vipnas"}'
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_x_samples = simcc_x_samples
        self.simcc_y_samples = simcc_y_samples

        num_deconv = len(deconv_out_channels) if deconv_out_channels else 0
        if num_deconv != 0:
            self.heatmap_size = tuple([s * (2**num_deconv) for s in in_featuremap_size])

            # deconv layers + 1x1 conv
            self.deconv_head = self._make_deconv_head(
                in_channels=in_channels,
                out_channels=out_channels,
                deconv_type=deconv_type,
                deconv_out_channels=deconv_out_channels,
                deconv_kernel_sizes=deconv_kernel_sizes,
                deconv_num_groups=deconv_num_groups,
                conv_out_channels=conv_out_channels,
                conv_kernel_sizes=conv_kernel_sizes,
                final_layer=final_layer,
            )

            if final_layer is not None:
                in_channels = out_channels
            else:
                in_channels = deconv_out_channels[-1]

        else:
            self.deconv_head = None

            if final_layer is not None:
                cfg = dict(
                    type="Conv2d",
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                )
                cfg.update(final_layer)
                self.final_layer = build_conv_layer(cfg)
            else:
                self.final_layer = None

            self.heatmap_size = in_featuremap_size

        # Define SimCC layers
        flatten_dims = self.heatmap_size[0] * self.heatmap_size[1]

        W = int(self.simcc_x_samples)
        H = int(self.simcc_y_samples)

        self.mlp_head_x = nn.Linear(flatten_dims, W)
        self.mlp_head_y = nn.Linear(flatten_dims, H)

    def _make_deconv_head(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        deconv_type: str = "heatmap",
        deconv_out_channels: OptIntSeq = (256, 256, 256),
        deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
        deconv_num_groups: OptIntSeq = (16, 16, 16),
        conv_out_channels: OptIntSeq = None,
        conv_kernel_sizes: OptIntSeq = None,
        final_layer: dict = dict(kernel_size=1),
    ) -> nn.Module:
        """Create deconvolutional layers by given parameters."""

        if deconv_type == "heatmap":
            deconv_head = MODELS.build(
                dict(
                    type="HeatmapHead",
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    deconv_out_channels=deconv_out_channels,
                    deconv_kernel_sizes=deconv_kernel_sizes,
                    conv_out_channels=conv_out_channels,
                    conv_kernel_sizes=conv_kernel_sizes,
                    final_layer=final_layer,
                )
            )
        else:
            deconv_head = MODELS.build(
                dict(
                    type="ViPNASHead",
                    in_channels=in_channels,
                    out_channels=out_channels,
                    deconv_out_channels=deconv_out_channels,
                    deconv_num_groups=deconv_num_groups,
                    conv_out_channels=conv_out_channels,
                    conv_kernel_sizes=conv_kernel_sizes,
                    final_layer=final_layer,
                )
            )

        return deconv_head

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward the network.

        The input is the featuremap extracted by backbone and the
        output is the simcc representation.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            pred_x (Tensor): 1d representation of x.
            pred_y (Tensor): 1d representation of y.
        """
        if self.deconv_head is None:
            feats = feats[-1]
            if self.final_layer is not None:
                feats = self.final_layer(feats)
        else:
            feats = self.deconv_head(feats)

        # flatten the output heatmap
        x = torch.flatten(feats, 2)

        pred_x = self.mlp_head_x(x)
        pred_y = self.mlp_head_y(x)

        return pred_x, pred_y

    def predict(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        test_cfg: OptConfigType = {},
    ) -> InstanceList:
        raise NotImplementedError("Do not support predict function")

    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: OptConfigType = {},
    ) -> dict:
        raise NotImplementedError("Do not support loss function")

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(type="Normal", layer=["Conv2d", "ConvTranspose2d"], std=0.001),
            dict(type="Constant", layer="BatchNorm2d", val=1),
            dict(type="Normal", layer=["Linear"], std=0.01, bias=0),
        ]
        return init_cfg
