import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import List, Any
from einops import rearrange
from ..utils.decode import CTCDecoder
from hydra.utils import instantiate

import lightning as L
from omegaconf.dictconfig import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from csi_sign_language.data_utils.ph14.post_process import post_process
from csi_sign_language.modules.losses.loss import VACLoss as _VACLoss
from csi_sign_language.data_utils.ph14.wer_evaluation_python import wer_calculation

from typing import List
from ..data_utils.interface_post_process import IPostProcess


class SLRModel(L.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        vocab,
        ctc_search_type="beam",
        file_logger=None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False, ignore=["cfg", "file_logger"])

        self.cfg = cfg
        self.data_excluded = getattr(cfg, "data_excluded", [])
        self.backbone: nn.Module = instantiate(cfg.model)

        self.loss = instantiate(cfg.loss)

        self.vocab = vocab
        self.decoder = CTCDecoder(
            vocab, blank_id=0, search_mode=ctc_search_type, log_probs_input=True
        )

        self.post_process = None

        self.train_ids_epoch = []
        self.val_ids_epoch = []
        # self.backbone.register_backward_hook(se()lf.check_gradients)

    @torch.no_grad()
    def _outputs2labels(self, out, length):
        # [t n c]
        # return list(list(string))
        y_predict = torch.nn.functional.log_softmax(out, -1).detach().cpu()
        return self.decoder(y_predict, length)

    @staticmethod
    def _gloss2sentence(gloss: List[List[str]]):
        return [" ".join(g) for g in gloss]

    @staticmethod
    def _extract_batch(batch):
        video = batch["video"]
        gloss = batch["gloss"]
        video_length = batch["video_length"]
        gloss_length = batch["gloss_length"]
        gloss_gt = batch["gloss_label"]
        id = batch["id"]
        return id, video, gloss, video_length, gloss_length, gloss_gt

    @staticmethod
    def check_gradients(module, grad_input, grad_output):
        for grad in grad_input:
            if grad is not None:
                if torch.isnan(grad).any():
                    print("graident is nan", file=sys.stderr)
                if torch.isinf(grad).any():
                    print(grad)
                    print("graident is inf", file=sys.stderr)

    def set_post_process(self, fn):
        self.post_process: IPostProcess = fn

    def forward(self, x, t_length) -> Any:
        outputs = self.backbone(x, t_length)
        hyp = self._outputs2labels(outputs.out, outputs.t_length)
        return outputs, hyp

    def predict_step(self, batch, batch_id) -> Any:
        id, video, gloss, video_length, gloss_length, gloss_gt = self._extract_batch(
            batch
        )
        with torch.inference_mode():
            outputs = self.backbone(video, video_length)
            hyp = self._outputs2labels(outputs.out, outputs.t_length)
        # [(id, hyp, gloss_gt), ...]
        return list(zip(id, hyp, gloss_gt))

    def on_train_epoch_start(self):
        self.train_ids_epoch.clear()

    def training_step(self, batch, batch_idx):
        id, video, gloss, video_length, gloss_length, gloss_gt = self._extract_batch(
            batch
        )
        B = len(id)

        try:
            outputs = self.backbone(video, video_length)
            loss = self.loss(outputs, video, video_length, gloss, gloss_length)
        except torch.cuda.OutOfMemoryError as e:
            print(f"cuda out of memory! the t_length is {video.size(2)}")
            raise e

        # if we should skip this batch
        skip_flag = torch.tensor(0, dtype=torch.uint8, device=self.device)
        if any(i in self.data_excluded for i in id):
            skip_flag = torch.tensor(1, dtype=torch.uint8, device=self.device)
        if torch.isnan(loss) or torch.isinf(loss):
            self.print(
                f"find nan, data id={id}, output length={outputs.t_length.cpu().numpy()}, label_length={gloss_length.cpu().numpy()}",
                file=sys.stderr,
            )
            skip_flag = torch.tensor(1, dtype=torch.uint8, device=self.device)
        flags = self.all_gather(skip_flag)
        if (flags > 0).any().item():
            del outputs
            del loss
            self.print(flags)
            self.print("skipped", file=sys.stderr)
            return

        hyp = self._outputs2labels(outputs.out.detach(), outputs.t_length.detach())

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=B,
        )
        self.log(
            "train_wer",
            wer_calculation(gloss_gt, hyp),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        self.train_ids_epoch += id
        return loss

    def on_validation_start(self) -> None:
        self.val_ids_epoch.clear()

    def validation_step(self, batch, batch_idx):
        id, video, gloss, video_length, gloss_length, gloss_gt = self._extract_batch(
            batch
        )
        B = len(id)

        with torch.inference_mode():
            outputs = self.backbone(video, video_length)
            loss = self.loss(outputs, video, video_length, gloss, gloss_length)

        hyp = self._outputs2labels(outputs.out, outputs.t_length)
        if self.post_process:
            hyp, gt = self.post_process.process(hyp, gloss_gt)
        else:
            raise NotImplementedError()

        self.log(
            "val_loss",
            loss.detach(),
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            batch_size=B,
        )
        self.log(
            "val_wer",
            wer_calculation(gt, hyp),
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            batch_size=B,
        )
        self.val_ids_epoch += id

    # def on_save_checkpoint(self, checkpoint: torch.Dict[str, Any]) -> None:
    #     params: dict = checkpoint['state_dict']
    #     for key in params.keys():
    #         if re.match('loss' )

    def configure_optimizers(self):
        opt: Optimizer = instantiate(
            self.cfg.optimizer,
            filter(lambda p: p.requires_grad, self.backbone.parameters()),
        )
        scheduler = instantiate(self.cfg.lr_scheduler, opt)
        return {"optimizer": opt, "lr_scheduler": scheduler}


class VACLoss(nn.Module):
    def __init__(self, weights, temp, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weights = weights
        self.loss = _VACLoss(weights, temp)

    def forward(self, outputs, input, input_length, target, target_length):
        if self.weights[2] > 0.0 or self.weights[1] > 0.0:
            conv_out = outputs.neck_out.out
            conv_length = outputs.neck_out.t_length
        else:
            conv_out = None
            conv_length = None

        seq_out = outputs.out
        t_length = outputs.t_length

        return self.loss(
            conv_out, conv_length, seq_out, t_length, target, target_length
        )
