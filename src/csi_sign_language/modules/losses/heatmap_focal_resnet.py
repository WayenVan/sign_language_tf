if __name__ == "__main__":
    import sys

    sys.path.append("/root/projects/sign_language_transformer/src")

import torch
import numpy as np
from torch import nn
from mmengine.config import Config
from mmengine.registry.utils import init_default_scope
from mmpose.apis import init_model
from csi_sign_language.utils.data import mapping_0_1
from einops import rearrange, repeat
from torch import Tensor
import torch.nn.functional as F
from mmpose.models import TopdownPoseEstimator
from torchvision import transforms
from torchvision.transforms.functional import normalize


def focal_heatmap_loss(a, h, gamma=2.0, epsilon=1e-8):
    assert (a >= 0).all(), f"a should be in [0, 1], but got min value: {a.min().item()}"
    assert (a <= 1).all(), f"a should be in [0, 1], but got max value: {a.max().item()}"
    assert (
        h >= 0
    ).all(), f"heatmap should be in [0, 1], but got min value: {h.min().item()}"
    assert (
        h <= 1
    ).all(), f"heatmap should be in [0, 1], but got max value: {h.max().item()}"

    return -((1 - h) ** gamma) * torch.log(1 - a + epsilon)


class Constants:
    left_hand = list(range(112, 133))
    righ_hand = list(range(91, 112))
    # body = list(range(0, 17))
    body = [0, 1, 2, 3, 4, 9, 10]
    face = list(range(23, 91))

    left_hand_center_ref = [121, 112]
    right_hand_center_ref = [100, 91]


# decode x, y to keypoints
def decode(x, y, simcc_split_ratio):
    x_locs = x.argmax(dim=-1)
    y_locs = y.argmax(dim=-1)
    locs = torch.stack([x_locs, y_locs], dim=-1).to(x.dtype)
    locs /= simcc_split_ratio
    return locs


@torch.no_grad()
def gaussian_heatmap(
    height,
    width,
    target_height,
    target_width,
    center_x: Tensor,
    center_y: Tensor,
    sigma,
    device="cuda:1",
):
    x_coords = torch.arange(0, target_width).float().to(device)
    y_coords = torch.arange(0, target_height).float().to(device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
    y_grid, x_grid = (
        repeat(
            a,
            "h w -> () h w",
        )
        for a in (y_grid, x_grid)
    )

    # rescale center_x and center_y to target size
    center_x = center_x * target_width / width
    center_y = center_y * target_height / height
    center_x, center_y = (
        repeat(a, "n -> n () ()").to(device) for a in (center_x, center_y)
    )

    # 计算高斯分布
    gaussian = torch.exp(
        -((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / (2 * sigma**2)
    )
    gaussian = gaussian / gaussian.max()
    # [n h w]
    return gaussian


class PadResize:
    def __init__(self, input_size, target_size) -> None:
        # align with width
        self.input_h, self.input_w = tuple(input_size)
        self.target_h, self.target_w = tuple(target_size)
        self.ratio = self.target_w / self.input_w

        self.resized_h = int(self.input_h * self.ratio)
        self.resized_w = self.target_w

        self.padding_h = self.target_h - self.resized_h
        assert self.padding_h >= 0

        self.resize = transforms.Resize(
            (self.resized_h, self.resized_w), transforms.InterpolationMode.BILINEAR
        )
        self.padding = transforms.Pad((0, 0, 0, self.padding_h))

    def resize_pad(self, image):
        _, _, H, W = image.shape
        assert H == self.input_h and W == self.input_w
        image = self.resize(image)
        return self.padding(image), self.resized_h, self.resized_w


class HeatmapFocalResnetLoss(nn.Module):
    model: TopdownPoseEstimator

    def __init__(
        self,
        dw_pose_cfg,
        dw_pose_ckpt,
        dw_pose_input_size,
        input_size,
        weights,
        sigmas,
        num_stages,
        stage_weights=None,
        gamma=2,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        assert (
            len(sigmas) == num_stages
        ), f"number of sigmas should be equal to number of stages, but got {len(sigmas)} and {num_stages}"

        cfg = Config.fromfile(dw_pose_cfg)
        init_default_scope("mmpose")
        self.model = init_model(cfg, dw_pose_ckpt, device="cpu")
        self.pad_resize = PadResize(input_size, dw_pose_input_size)

        self.mean = cfg.model.data_preprocessor.mean
        self.std = cfg.model.data_preprocessor.std
        self.gamma = gamma

        if stage_weights is None:
            self.stage_weights = [1.0] * num_stages
        else:
            assert len(stage_weights) == num_stages
            self.stage_weights = stage_weights

        self.sigmas = sigmas
        self.ctc_weight, self.heatmap_weight = tuple(weights)
        self._loss_ctc = nn.CTCLoss(blank=0, reduction="none")
        self._freeze_pose_model()

    def _freeze_pose_model(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def _pre_norm(self, x):
        return normalize(x, self.mean, self.std)

    def forward(self, outputs, input, input_length, target, target_length):
        loss = 0.0
        if self.ctc_weight > 0.0:
            out = nn.functional.log_softmax(outputs.out, dim=-1)
            ctc_loss = self._loss_ctc(
                out, target, outputs.t_length.cpu().int(), target_length.cpu().int()
            ).mean()
            loss += self.ctc_weight * ctc_loss

        if self.heatmap_weight > 0.0:
            heatmap_loss, _ = self._loss_heatmap(
                outputs.encoder_out.attn_weights, input
            )
            loss += self.heatmap_weight * heatmap_loss
        return loss

    def _loss_heatmap(self, attention_maps, input):
        # attention maps: tuple of [t b heads hands&face h w]
        # input: [n c t h w]
        attention_maps = list(
            rearrange(a, "t b heads fh h w -> (b t) heads fh h w")
            for a in attention_maps
        )
        input = rearrange(input, "b c t h w -> (b t) c h w")

        loss = 0.0
        with torch.no_grad():
            # n s h w
            kps, h, w = self._keypoints(input)
            target_heatmaps = []
            for i in range(len(attention_maps)):
                target_h, target_w = attention_maps[i].shape[-2:]
                sigma = self.sigmas[i]
                # b fh h w
                target_heatmap = self.draw_gaussian(
                    kps, (h, w), (target_h, target_w), sigma
                )
                target_heatmaps.append(target_heatmap)
                target_heatmap = rearrange(target_heatmap, "b fh h w -> b () fh h w")
                # b heads fh h w
                with torch.enable_grad():
                    _loss = focal_heatmap_loss(
                        attention_maps[i], target_heatmap, self.gamma
                    )
                    _loss = torch.mean(_loss)
                    loss += _loss * self.stage_weights[i]

        return loss, target_heatmaps

    @torch.no_grad()
    def _keypoints(self, input):
        # input [n c h w]
        N, _, _, _ = input.shape
        input, valid_h, valid_w = self.pad_resize.resize_pad(input)
        input = self._pre_norm(input)
        x, y = self.model(input, None, "tensor")
        kps = decode(x, y, self.model.head.simcc_split_ratio)

        def append_centerpoints(predicted_points, centers):
            center_point = predicted_points[:, centers, :].mean(dim=1, keepdim=True)
            return torch.cat([predicted_points, center_point], dim=1)

        # add a new center point for left hand and right hand
        kps = append_centerpoints(kps, Constants.left_hand_center_ref)
        kps = append_centerpoints(kps, Constants.right_hand_center_ref)

        return kps, valid_h, valid_w

    @torch.no_grad()
    def draw_gaussian(self, kps, image_space, target_size, sigma):
        origin_h, origin_w = image_space
        target_h, target_w = target_size

        N, _, _ = kps.shape

        kps_hands = kps[:, Constants.left_hand + Constants.righ_hand + [-2, -1]]
        kps_face = kps[:, Constants.face]

        _, length_hands, _ = kps_hands.shape
        _, length_face, _ = kps_face.shape

        kps = torch.cat([kps_hands, kps_face], dim=1)

        # generate gaussian heatmap
        kps = rearrange(kps, "n k yx -> (n k) yx")
        gmap = gaussian_heatmap(
            origin_h,
            origin_w,
            target_h,
            target_w,
            kps[:, 0],
            kps[:, 1],
            sigma,
            str(kps.device),
        )
        gmap = rearrange(gmap, "(n k) h w -> n k h w", n=N)

        gmap_hands = gmap[:, :length_hands].max(dim=1)[0]
        gmap_face = gmap[:, length_hands:].max(dim=1)[0]

        # b fh h w
        return torch.stack([gmap_face, gmap_hands], dim=1)

    def _save_heatmap(self, heatmap, path):
        # n s h w
        heatmap = heatmap[0]
        import matplotlib.pyplot as plt
        import os

        for i in range(heatmap.shape[0]):
            file_name = f"heatmap_stage_{i}.jpg"
            fpath = os.path.join(path, file_name)
            h = heatmap[i].cpu().numpy()
            plt.imshow(h, cmap="hot", interpolation="nearest")
            plt.savefig(fpath)


if __name__ == "__main__":
    ckpt = "/root/projects/sign_language_transformer/resources/dwpose-l/dw-ll_ucoco.pth"
    cfg = "/root/projects/sign_language_transformer/resources/dwpose-l/rtmpose-l_8xb64-270e_coco-ubody-wholebody-256x192.py"

    loss = HeatmapFocalResnetLoss(
        dw_pose_cfg=cfg,
        dw_pose_ckpt=ckpt,
        dw_pose_input_size=(256, 192),
        input_size=(224, 224),
        weights=[1.0, 1.0],
        sigmas=[1, 1.0, 1.0],
        num_stages=3,
    ).to("cuda:1")

    image = "../resources/test_image.png"
    cfg = Config.fromfile(cfg)

    # prepare data
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    def central_crop(image, crop_width, crop_height):
        # Get the dimensions of the image
        img_width, img_height = image.size

        # Calculate the coordinates for the central crop
        left = (img_width - crop_width) / 2
        top = (img_height - crop_height) / 2
        right = (img_width + crop_width) / 2
        bottom = (img_height + crop_height) / 2

        # Perform the crop
        cropped_image = image.crop((left, top, right, bottom))

        return cropped_image

    image = Image.open("./resources/test_image.png")
    image = central_crop(image, 224, 224)
    print(f"data type: {np.array(image).dtype}")

    data = torch.from_numpy(np.array(image)).float()
    print(f"std: {data.std()}, mean: {data.mean()}")

    data = rearrange(data, " h w c ->() c h w")
    data = repeat(data, " n c h w -> n c t h w", t=100)
    data = data.to("cuda:1")

    attention = [
        torch.randn(100, 1, 12, 2, 12, 12).to("cuda:1"),
        torch.randn(100, 1, 12, 2, 7, 7).to("cuda:1"),
        torch.randn(100, 1, 12, 2, 4, 4).to("cuda:1"),
    ]

    def transform(a: Tensor):
        H = a.shape[-2]
        a = rearrange(a, "t b heads fh h w -> t b heads fh (h w)")
        a = F.softmax(a, dim=-1)
        a = rearrange(a, "t b heads fh (h w) -> t b heads fh h w", h=H)
        return a

    attention = list(transform(a) for a in attention)
    loss, heatmaps = loss._loss_heatmap(attention, data)
    print(loss)

    def blend_images(base_img, overlay_img, alpha=0.5):
        """Blend the base image with the overlay iMage."""
        return (1 - alpha) * base_img + alpha * overlay_img

    origin = np.array(image) / 255.0

    for stage in range(3):
        plt.clf()
        h = heatmaps[stage][0, 1, :].cpu().numpy()
        plt.imshow(h, cmap="hot", interpolation="nearest")
        plt.colorbar()
        plt.savefig("outputs/heatmap_stage{}.jpg".format(stage))
        import cv2

        h = cv2.resize(h, (224, 224))

        def apply_colormap(heatmap):
            """Apply a colormap to the heatmap and normalize."""
            cmap = plt.get_cmap("jet")
            heatmap_colored = cmap(heatmap)  # Apply colormap
            return heatmap_colored[..., :3]  # Discard alpha channel

        h = apply_colormap(h)
        blend_image = blend_images(origin, h, alpha=0.6)
        import matplotlib.pyplot as plt

        plt.imshow(blend_image)
        plt.savefig("outputs/blend_image_stage{}.jpg".format(stage))
