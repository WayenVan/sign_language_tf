import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import List, Any
from einops import rearrange
from ..utils.decode import CTCDecoder
from ..modules.base_stream import BaseStream
from collections import namedtuple
from hydra.utils import instantiate

import lightning as L
from omegaconf.dictconfig import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from csi_sign_language.data_utils.ph14.post_process import post_process
from csi_sign_language.modules.loss import VACLoss as _VACLoss
from csi_sign_language.modules.loss import HeatMapLoss
from csi_sign_language.data_utils.ph14.wer_evaluation_python import wer_calculation

from torchmetrics.text import WordErrorRate
from typing import List
from ..data_utils.interface_post_process import IPostProcess
import re



class SLRModel(L.LightningModule):

    def __init__(
        self, 
        cfg: DictConfig,
        vocab,
        ctc_search_type = 'beam',
        file_logger = None,
         *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False, ignore=['cfg', 'file_logger'])

        self.cfg = cfg
        self.data_excluded = getattr(cfg, 'data_excluded', [])
        self.backbone: nn.Module = instantiate(cfg.model)
        self.loss: nn.Module = instantiate(cfg.loss)

        self.vocab = vocab
        self.decoder = CTCDecoder(vocab, blank_id=0, search_mode=ctc_search_type, log_probs_input=True)

        self.post_process = None

        self.train_ids_epoch = []
        self.val_ids_epoch = []
        self.backbone.register_backward_hook(self.check_gradients)
        
    @torch.no_grad()
    def _outputs2labels(self, out, length):
        #[t n c]
        #return list(list(string))
        y_predict = torch.nn.functional.log_softmax(out, -1).detach().cpu()
        return self.decoder(y_predict, length)

    @staticmethod
    def _gloss2sentence(gloss: List[List[str]]):
        return [' '.join(g) for g in gloss]
    
    @staticmethod
    def _extract_batch(batch):
        video = batch['video']
        gloss = batch['gloss']
        video_length = batch['video_length']
        gloss_length = batch['gloss_length']
        gloss_gt = batch['gloss_label']
        id = batch['id']
        return id, video, gloss, video_length, gloss_length, gloss_gt
    
    @staticmethod
    def check_gradients(module, grad_input, grad_output):
        for grad in grad_input:
            if grad is not None:
                if torch.isnan(grad).any():
                    print('graident is nan', file=sys.stderr)
                if torch.isinf(grad).any():
                    print(grad)
                    print('graident is inf', file=sys.stderr)

    
    def on_save_checkpoint(self, checkpoint: torch.Dict[str, Any]) -> None:
        #remove any parameters in loss function to reduce thesize
        state_dict = checkpoint['state_dict']
        state_dict = {k: v for k, v in state_dict.items() if not re.match(r'^loss', k)}
        checkpoint['state_dict'] = state_dict
        
    def set_post_process(self, fn):
        self.post_process: IPostProcess = fn

    def forward(self, x, t_length) -> Any:
        return self.backbone(x, t_length)
    
    def predict_step(self, batch, batch_id) -> Any:
        id, video, gloss, video_length, gloss_length, gloss_gt = self._extract_batch(batch)
        with torch.inference_mode():
            outputs = self.backbone(video, video_length)
            hyp = self._outputs2labels(outputs.out, outputs.t_length)
        #[(id, hyp, gloss_gt), ...]
        return list(zip(id, hyp, gloss_gt))

    def on_train_epoch_start(self):
        self.train_ids_epoch.clear()

    def training_step(self, batch, batch_idx):
        id, video, gloss, video_length, gloss_length, gloss_gt = self._extract_batch(batch)

        outputs = self.backbone(video, video_length)
        loss = self.loss(outputs, video, video_length, gloss, gloss_length)
        
        # if we should skip this batch
        skip_flag = torch.tensor(0, dtype=torch.uint8, device=self.device)
        if any(i in self.data_excluded for i in id):
            skip_flag = torch.tensor(1, dtype=torch.uint8, device=self.device)
        if torch.isnan(loss) or torch.isinf(loss):
            self.print(f'find nan, data id {id}', file=sys.stderr)
            skip_flag = torch.tensor(1, dtype=torch.uint8, device=self.device)
        flags = self.all_gather(skip_flag)
        if (flags > 0).any().item():
            del outputs
            del loss
            self.print(flags)
            self.print('skipped', file=sys.stderr)
            return 

        hyp = self._outputs2labels(outputs.out.detach(), outputs.t_length.detach())

        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log('train_wer', wer_calculation(gloss_gt, hyp), on_step=False, on_epoch=True, sync_dist=True)
        self.train_ids_epoch += id
        return loss
    
    def on_validation_start(self) -> None:
        self.val_ids_epoch.clear()

    def validation_step(self, batch, batch_idx):
        id, video, gloss, video_length, gloss_length, gloss_gt = self._extract_batch(batch)

        with torch.inference_mode():
            outputs = self.backbone(video, video_length)
            loss = self.loss(outputs, video, video_length, gloss, gloss_length)

        hyp = self._outputs2labels(outputs.out, outputs.t_length)
        if self.post_process:
            hyp, gt= self.post_process.process(hyp, gloss_gt)
        else:
            raise NotImplementedError()

        self.log('val_loss', loss.detach(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('val_wer', wer_calculation(gt, hyp), on_epoch=True, on_step=False, sync_dist=True)
        self.val_ids_epoch += id
    

    def configure_optimizers(self):
        opt: Optimizer = instantiate(self.cfg.optimizer, filter(lambda p: p.requires_grad, self.backbone.parameters()))
        scheduler = instantiate(self.cfg.lr_scheduler, opt)
        return {
            'optimizer': opt,
            'lr_scheduler': scheduler
        }
    
class VACLoss(nn.Module):

    def __init__(self, weights, temp, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss = _VACLoss(weights, temp)

    def forward(self, outputs, input, input_length, target, target_length): 
        conv_out = outputs.encoder_out.out
        conv_length = outputs.encoder_out.t_length
        seq_out = outputs.out
        t_length = outputs.t_length
        return self.loss(conv_out, conv_length, seq_out, t_length, target, target_length)

class MultiLoss(nn.Module):
    
    def __init__(
        self,
        weights,
        color_range,
        cfg,
        ckpt) -> None:
        super().__init__()
        self.weights = weights
        self.pose_loss = HeatMapLoss(color_range, cfg, ckpt)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='none')
    
    def forward(self, outputs, input, input_length, target, target_length): 
        #n c t h w
        heatmap_out = outputs.backbone_out.encoder_out.heatmap
        out = F.log_softmax(outputs.backbone_out.out, dim=-1)
        t_length = outputs.backbone_out.t_length
        loss = 0.
        
        if self.weights[0] > 0.:
            loss += self.ctc_loss(out, target.cpu().int(), t_length.cpu().int(), target_length.cpu().int()).mean()* self.weights[0]
        if self.weights[1] > 0.:
            heatmap_out = rearrange(heatmap_out, 'n c t h w -> (n t) c h w')
            input_ = rearrange(input, 'n c t h w -> (n t) c h w')
            loss += self.pose_loss(heatmap_out, input_) * self.weights[1]
        return loss 