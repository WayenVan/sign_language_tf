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

class Constants:
    left_hand = list(range(112, 133))
    righ_hand = list(range(91, 112))
    body = list(range(0, 17))
    face = list(range(23, 91))

    left_hand_center_ref = [121, 112]
    right_hand_center_ref = [100, 91]

@torch.no_grad()
def gaussian_heatmap(height, width, center_x: Tensor, center_y: Tensor, sigma: Tensor, device='cuda:1'):
    # sigma [stage]
    # [n ]
    x_coords = torch.arange(0, width).float().to(device)
    y_coords = torch.arange(0, height).float().to(device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    y_grid, x_grid = (repeat(a, 'h w -> n s h w', n = 1, s=1).contiguous() for a in (y_grid, x_grid))
    center_x, center_y = (repeat(a, 'n -> n s h w', s=1, h=1, w=1).contiguous().to(device) for a in (center_x, center_y))
    sigma = repeat(sigma, 's -> n s h w', n=1, h=1, w=1).to(device).contiguous()

    # 计算高斯分布
    gaussian = torch.exp(-((x_grid - center_x)**2 + (y_grid - center_y)**2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.max()
    #[n s h w]
    return gaussian



class PadResize:
    def __init__(self, input_size, target_size) -> None:
        #align with width
        self.input_h, self.input_w = tuple(input_size)
        self.target_h, self.target_w = tuple(target_size)
        self.ratio = self.target_w / self.input_w

        self.resized_h = self.input_h * self.ratio
        self.resized_w = self.target_w
        
        self.padding_h = self.target_h - self.resized_h
        assert self.padding_h >= 0

        self.resize = transforms.Resize((self.resized_h, self.resized_w), transforms.InterpolationMode.BILINEAR)
        self.padding = transforms.Pad((0, 0, 0, self.padding_h))

    def resize_pad(self, image):
        _, _, H, W = image.shape
        assert H == self.input_h and W == self.input_w
        image = self.resize(image)
        return self.padding(image)
    
    def remove_padding(self, image):
        _, _, H, W = image.shape
        assert H == self.target_h, W == self.target_w
        return image[..., :-self.padding_h, :]
        
    
class HeatMapLoss(nn.Module):

    def __init__(
        self, 
        dw_pose_cfg,
        dw_pose_ckpt,
        dw_pose_input_size,
        input_size,
        sigmas,
        stage_lambda,
         *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        cfg = Config.fromfile(cfg)
        init_default_scope('mmpose')
        self.model: TopdownPoseEstimator = init_model(dw_pose_cfg, dw_pose_ckpt, device='cpu')
        self.pad_resize = PadResize(input_size, dw_pose_input_size)
        self.n_padding_h = self.pad_resize.padding_h
        self.dw_pose_input_size = dw_pose_input_size

        self.sigmas = sigmas
        self.stage_lambda =stage_lambda


    def forward(self, outputs, input, input_length, target, target_length): 
        ctc_loss = self.CTC(outputs.out, target, outputs.t_length.cpu().int(), target_length.cpu().int()).mean()
        heatmap_loss = self._loss_heatmap(None, input)
        return ctc_loss + heatmap_loss

    def _loss_heatmap(self, predicted_heatmap_logits, input):
        # predicted_heatmap_logits t n s 2 h w
        # stage_lambda [s]
        # input: [n c t h w]
        predicted_heatmap_logits = rearrange(predicted_heatmap_logits, 't n s 2 h w -> (n t) s 2 h w')
        input = rearrange(input, 'n c t h w -> (n t) c h w')

        stage_lambda = torch.tensor(self.stage_lambda, device=input.device, dtype=input.dtype).to(input.device)
        _, S1, _, H1, W1= predicted_heatmap_logits.shape

        # n s h w
        predicted_heatmap = nn.functional.gumbel_softmax(predicted_heatmap_logits, dim=-1, hard=True)[:, :, 1, :, :]
        # n s h w
        target_heatmap = self._keypoints(input)
        _, S2, H2, W2 = target_heatmap.shape

        assert S1 == S2, "stage should be the same"
        assert H1 == H2, "height should be the same"
        assert W1 == W2, "width should be the same"

        loss = F.mse_loss(predicted_heatmap, target_heatmap, reduce=False)
        # [n s h w]
        loss, = torch.mean(loss, dim=[0, 2, 3])
        loss = torch.sum(loss * stage_lambda)
        return loss


    @torch.no_grad()
    def _keypoints(self, input):
        #input [n c h w]
        N, _, _, _ = input.shape
        input = self.pad_resize.resize_pad(input)
        y, x = self.model(input, None, 'tensor')

        # n k
        keypoints = self.model.head.decode((y, x))
        kps = torch.stack(list(torch.from_numpy(keypoint['keypoints'][0]) for keypoint in keypoints))
        def append_centerpoints(predicted_points, centers):
            center_point = predicted_points[:, centers, :].mean(dim=1, keepdim=True)
            return torch.cat([predicted_points, center_point], dim=1)

        #add a new center point for left hand and right hand
        kps = append_centerpoints(kps, Constants.left_hand_center)
        kps = append_centerpoints(kps, Constants.right_hand_center)

        #generate gaussian heatmap
        kps = rearrange(kps, 'n k xy -> (n k) xy')
        gmap = gaussian_heatmap(self.dw_pose_input_size[0], self.dw_pose_input_size[1], kps[:, 0], kps[:, 1], self.sigmas, input.device)
        gmap = rearrange(gmap, '(n k) s h w -> n k s h w', n=N)
        gmap = gmap[:, Constants.body + Constants.face + Constants.left_hand + Constants.righ_hand]
        gmap, = torch.max(gmap, dim=1, keepdim=False)
        gmap = self.pad_resize.remove_padding(gmap)
        self._save_heatmap(gmap, 'outputs/heatmap')
        # [n s h w]
        return gmap
        
    def _save_heatmap(self, heatmap, path):
        # n s h w
        heatmap = heatmap[0]
        import cv2
        import os
        for i in range(heatmap.shape[0]):
            file_name = f"heatmap_stage_{i}.jpg"
            path = os.path.join(path, file_name)
            h = heatmap[i].cpu().numpy()
            h = mapping_0_1(h)
            cv2.imwrite(path, h)

if __name__ == '__main__':
    ckpt = '/root/projects/sign_language_transformer/resources/dwpose-l/dw-ll_ucoco.pth'
    cfg = '/root/projects/sign_language_transformer/resources/dwpose-l/rtmpose-l_8xb64-270e_coco-ubody-wholebody-256x192.py'

    loss = HeatMapLoss(
        dw_pose_cfg=ckpt,
        dw_pose_ckpt=cfg,
        dw_pose_input_size=(256, 192),
        input_size=(224, 224),
        sigmas=[20, 15, 10, 5],
        stage_lambda=[1.0, 1.0, 1.0, 1.0],
    )
    
    image = '../resources/test_image.png'
    cfg = Config.fromfile(cfg)
    std = torch.tensor(cfg.model.data_preprocessor.std)
    mean = torch.tensor(cfg.model.data_preprocessor.mean)
    
    #prepare data
    from PIL import Image
    import numpy as np

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

    image = Image.open('../resources/test_image.png')
    image = central_crop(image, 224, 224)
    print(f'data type: {np.array(image).dtype}')

    data = torch.from_numpy(np.array(image)).float()
    data = (data - std) / mean
    print(f'std: {data.std()}, mean: {data.mean()}')

    data = rearrange(data, ' h w c ->() c h w')
    data = repeat(data, ' n c h w -> n c t h w', t=10)
    data.to('cuda:1')
    
    logits = torch.randn(10, 1, 4, 2, 224, 224).to('cuda:1')
    loss._loss_heatmap(logits, data, [1.0, 1.0, 1.0, 1.0])