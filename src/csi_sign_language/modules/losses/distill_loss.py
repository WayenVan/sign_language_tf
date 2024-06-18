
import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from ...models.slr_model import SLRModel
from omegaconf import OmegaConf
from .loss import VACLoss
from mmengine.registry.utils import init_default_scope

class VacDistLoss(nn.Module):

    def __init__(
        self, 
        alpha, 
        beta, 
        sigma, 
        tau, 
        checkpoint,
        cfg,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.vacloss = VACLoss([alpha, beta, 0.], tau)
        self.distill = DistillLoss(checkpoint, cfg, tau)
        self.sigma = sigma

    def forward(self, outputs, input, input_length, target, target_length): 
        conv_out = outputs.neck_out.out
        conv_length = outputs.neck_out.t_length

        seq_out = outputs.out
        t_length = outputs.t_length

        ctc_loss = self.vacloss(conv_out, conv_length, seq_out, t_length, target, target_length)
        distill = self.distill(input, input_length, conv_out)
        return ctc_loss + self.sigma * distill

class DistillLoss(nn.Module):
    def __init__(
        self, 
        checkpoint,
        cfg,
        tau,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg = OmegaConf.load(cfg)
        init_default_scope('mmpretrain')
        model_temp = SLRModel.load_from_checkpoint(checkpoint, map_location='cpu', cfg=cfg)

        self.backbone = model_temp.backbone
        self.backbone.eval()
        self.kl = nn.KLDivLoss(log_target=True, reduction='batchmean')
        self.tau = tau

        del model_temp
    
    def forward(self, input, input_length, neck_out):
        targets = self.get_target(input, input_length)
        T = self.tau

        #[t n c]
        neck_target = targets.neck_out.out
        loss = self.kl(
            F.log_softmax(neck_out/T),
            F.log_softmax(neck_target/T)
        ) * (T * T) 
        return loss
        
    @torch.no_grad()
    def get_target(self, input, input_length):
        return self.backbone(input, input_length)
