import torch
from torch import nn
from mmengine.config import Config
from mmengine.registry.utils import init_default_scope
from mmpose.apis import init_model
from csi_sign_language.utils.data import mapping_0_1
from einops import rearrange
import torch.nn.functional as F



class VACLoss:

    def __init__(self, weights, temperature) -> None:
        #weigts: ctc_seq, ctc_conv, distill
        self.CTC = nn.CTCLoss(blank=0, reduction='none')
        self.distll = SelfDistillLoss(temperature)
        self.weights = weights
    
    def __call__(self, conv_out, conv_length, seq_out, length, target, target_length):
        #[t, n, c] logits

        seq_out = F.log_softmax(seq_out, dim=-1)
        seq_out = seq_out.to(torch.float32)
        
        if self.weights[1] > 0. or self.weights[2] > 0.:
            conv_out = F.log_softmax(seq_out, dim=-1)
            conv_out = conv_out.to(torch.float32)

        loss = 0
        if self.weights[0] > 0.:
            loss += self.CTC(seq_out, target, length.cpu().int(), target_length.cpu().int()).mean()* self.weights[0]
        if self.weights[1] > 0.:
            loss += self.CTC(conv_out, target, conv_length.cpu().int(), target_length.cpu().int()).mean() * self.weights[1]
        if self.weights[2] > 0.:
            loss += self.distll(seq_out, conv_out) * self.weights[2]
        return loss    
    
    # def _filter_nan(self, *losses):
    #     ret = []
    #     for loss in losses:
    #         if torch.all(torch.isinf(loss)).item():
    #             loss: torch.Tensor
    #             print('loss is inf')
    #             loss = torch.nan_to_num(loss, posinf=0.)
    #         ret.append(loss)
    #     return tuple(ret)

class SelfDistillLoss:
    def __init__(self, temperature) -> None:
        self.t = temperature
        
    def __call__(self, teacher, student):
        # seq: logits [t, n, c]
        T, N, C = teacher.shape
        assert (T, N, C) == student.shape
        teacher, student = teacher/self.t, student/self.t

        teacher = F.log_softmax(rearrange(teacher, 't n c -> (t n) c'), dim=-1)
        student = F.log_softmax(rearrange(student, 't n c -> (t n) c'), dim=-1)
        return F.kl_div(student, teacher.detach(), log_target=True, reduction='batchmean')

class HeatMapLoss(nn.Module):
    
    def __init__(self, 
                color_range, 
                cfg_path, 
                checkpoint) -> None:
        super().__init__()
        cfg = Config.fromfile(cfg_path)

        self.cfg = cfg
        self.color_range = color_range

        vitpose = init_model(cfg, checkpoint, device='cpu')
        vitpose = vitpose.half()
        for p in vitpose.parameters():
            p.requires_grad = False
        vitpose.eval()
        
        #ignore the vitpsoe by hid it behind a list
        self.vitpose = [vitpose]

        self.register_buffer('std', torch.tensor(cfg.model.data_preprocessor.std))
        self.register_buffer('mean', torch.tensor(cfg.model.data_preprocessor.mean))
        
        self.l2_loss = nn.MSELoss(reduction='mean')
        
    
    def forward(self, heatmap, input):
        #n 17 t h w
        return self.l2_loss(heatmap, self._vitpose_predict(input))

    @torch.inference_mode()
    def _vitpose_predict(self, x):
        #n c h w
        assert x.min() >= self.color_range[0], f'{x.min()}'
        assert x.max() <= self.color_range[1]+0.1, f'{x.max()}'

        if self.is_cuda:
            x = x.half()

        x = self._data_preprocess(x)
        heatmap = self.vitpose[0](x, None)
        return heatmap.detach().clone()

    @torch.inference_mode()
    def _data_preprocess(self, x):
        x = mapping_0_1(self.color_range, x)
        x = x * 255. #mapping to 0-255
        x = x.permute(0, 2, 3, 1)
        x = (x - self.mean) / self.std
        x = x.permute(0, 3, 1, 2)
        return x