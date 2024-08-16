if __name__ == '__main__':
    import sys
    sys.path.append('/root/projects/sign_language_transformer/src')

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


def focal_heatmap_loss(a, h, gamma=2.0, epsilon=1e-8):

    assert (a >= 0 ).all(), f'a should be in [0, 1], but got min value: {a.min().item()}'
    assert (a <= 1 ).all(), f'a should be in [0, 1], but got max value: {a.max().item()}'
    assert (h >= 0 ).all(), f'heatmap should be in [0, 1], but got min value: {h.min().item()}'
    assert (h <= 1 ).all(), f'heatmap should be in [0, 1], but got max value: {h.max().item()}'

    return - (1 - h)**gamma * torch.log(1 - a + epsilon)
    
class Constants:
    left_hand = list(range(112, 133))
    righ_hand = list(range(91, 112))
    # body = list(range(0, 17))
    body = [0, 1, 2, 3, 4, 9, 10]
    face = list(range(23, 91))

    left_hand_center_ref = [121, 112]
    right_hand_center_ref = [100, 91]

# @torch.no_grad()
# def gaussian_heatmap(height, width, center_x: Tensor, center_y: Tensor, sigma: Tensor, device='cuda:1'):
#     # sigma [stage]
#     # [n ]
#     x_coords = torch.arange(0, width).float().to(device)
#     y_coords = torch.arange(0, height).float().to(device)
#     y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
#     y_grid, x_grid = (repeat(a, 'h w -> n s h w', n = 1, s=1).contiguous() for a in (y_grid, x_grid))
#     center_x, center_y = (repeat(a, 'n -> n s h w', s=1, h=1, w=1).contiguous().to(device) for a in (center_x, center_y))
#     sigma = repeat(sigma, 's -> n s h w', n=1, h=1, w=1).to(device).contiguous()

#     # 计算高斯分布
#     gaussian = torch.exp(-((x_grid - center_x)**2 + (y_grid - center_y)**2) / (2 * sigma**2))
#     gaussian = gaussian / gaussian.max()
#     #[n s h w]
#     return gaussian

#decode x, y to keypoints
def decode(x, y, simcc_split_ratio):
    x_locs = x.argmax(dim=-1)
    y_locs = y.argmax(dim=-1)
    locs = torch.stack([x_locs, y_locs], dim=-1).to(x.dtype)
    locs /= simcc_split_ratio
    return locs

@torch.no_grad()
def gaussian_heatmap(
    height, width, center_x: Tensor, center_y: Tensor, sigma: Tensor, target_height, target_width, device='cuda:1', ):
    # sigma [stage]
    # [n ]
    
    x_coords = torch.arange(0, target_width).float().to(device)
    y_coords = torch.arange(0, target_height).float().to(device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    y_grid, x_grid = (repeat(a, 'h w -> n s h w', n = 1, s=1).contiguous() for a in (y_grid, x_grid))
    
    center_x, center_y = (a * target_width / width for a in (center_x, center_y))
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

        self.resized_h = int(self.input_h * self.ratio)
        self.resized_w = self.target_w
        
        self.padding_h = self.target_h - self.resized_h
        assert self.padding_h >= 0

        self.resize = transforms.Resize((self.resized_h, self.resized_w), transforms.InterpolationMode.BILINEAR)
        self.padding = transforms.Pad((0, 0, 0, self.padding_h))

    def resize_pad(self, image):
        _, _, H, W = image.shape
        assert H == self.input_h and W == self.input_w
        image = self.resize(image)
        return self.padding(image), self.resized_h, self.resized_w
    
    # def remove_padding(self, image):
    #     _, _, H, W = image.shape
    #     assert H == self.target_h, W == self.target_w
    #     return image[..., :-self.padding_h, :]
        
    
class HeatmapFocalLoss(nn.Module):

    def __init__(
        self, 
        dw_pose_cfg,
        dw_pose_ckpt,
        dw_pose_input_size,
        input_size,
        weights,
        sigmas,
        stages,
        stage_lambda = None,
        gamma = 2,
         *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        assert len(sigmas) == len(stages), f"sigmas should have the same length with stages, but got {len(sigmas)} and {len(stages)}"

        # cfg = Config.fromfile(dw_pose_cfg)
        init_default_scope('mmpose')
        self.model: TopdownPoseEstimator = init_model(dw_pose_cfg, dw_pose_ckpt, device='cpu')
        self.pad_resize = PadResize(input_size, dw_pose_input_size)
        self.n_padding_h = self.pad_resize.padding_h
        self.dw_pose_input_size = dw_pose_input_size
        self.ctc_weight, self.heatmap_weight = tuple(weights)

        self.sigmas = nn.Parameter(torch.tensor(sigmas), requires_grad=False)
        self.gamma = gamma
        
        if stage_lambda is not None:
            self.stage_lambda =nn.Parameter(torch.tensor(stage_lambda), requires_grad=False)
        else:
            self.stage_lambda = nn.Parameter(torch.ones(len(stages), dtype=torch.float32), requires_grad=False)
            
        self.stages = tuple(sorted(stages, reverse=True))

        self._loss_ctc = nn.CTCLoss(blank=0, reduction='none')
        self._freeze_pose_model()
    
    def _freeze_pose_model(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, outputs, input, input_length, target, target_length): 
        loss = 0.0
        if self.ctc_weight > 0.:
            out = nn.functional.log_softmax(outputs.out, dim=-1)
            ctc_loss = self._loss_ctc(out, target, outputs.t_length.cpu().int(), target_length.cpu().int()).mean()
            loss += self.ctc_weight * ctc_loss

        if self.heatmap_weight > 0.:
            heatmap_loss, _ = self._loss_heatmap(outputs.encoder_out.attn_weights, input)
            loss += self.heatmap_weight * heatmap_loss
        return loss

    def _loss_heatmap(self, attention_maps, input):
        # attention maps: [t n s head keys h w]
        # input: [n c t h w]
        # stage_lambda [s]
        attention_maps = attention_maps[:, :, self.stages]
        attention_maps = rearrange(attention_maps, 't n s heads keys h w -> (n t) s heads keys h w')
        input = rearrange(input, 'n c t h w -> (n t) c h w')

        _, S1, _, _, H1, W1 = attention_maps.shape

        
        with torch.no_grad():
            # n s h w
            target_heatmap = self._keypoints(input, H1, W1)
            _, S2, H2, W2 = target_heatmap.shape
            # self._save_heatmap(target_heatmap, 'outputs/target_heatmap')
            target_heatmap_ret = target_heatmap

        assert S1 == S2, f"stage should be the same, but got {S1} and {S2}"
        assert H1 == H2, f"height should be the same, but got {H1} and {H2}"
        assert W1 == W2, f"width should be the same, but got {W1} and {W2}"
        
        # n s heas key h w
        target_heatmap = rearrange(target_heatmap, 'n s h w -> n s () () h w')
        loss = focal_heatmap_loss(attention_maps, target_heatmap, gamma=self.gamma)
        # n s heads key h w
        loss = torch.sum(loss, dim=[-1, -2, -3])
        # n s heads
        loss = torch.mean(loss, dim=[0, 2])
        loss = torch.sum(loss * self.stage_lambda)
        return loss, target_heatmap_ret

    @torch.no_grad()
    def _keypoints(self, input, patch_height, patch_width):
        #input [n c h w]
        N, _, _, _ = input.shape
        input, valid_h, valid_w = self.pad_resize.resize_pad(input)
        x, y = self.model(input, None, 'tensor')

        # n k
        # keypoints = self.model.head.decode((x, y))
        # kps = torch.stack(list(torch.from_numpy(keypoint['keypoints'][0]) for keypoint in keypoints))
        kps = decode(x, y, self.model.head.simcc_split_ratio)
        def append_centerpoints(predicted_points, centers):
            center_point = predicted_points[:, centers, :].mean(dim=1, keepdim=True)
            return torch.cat([predicted_points, center_point], dim=1)
        
        #add a new center point for left hand and right hand
        kps = append_centerpoints(kps, Constants.left_hand_center_ref)
        kps = append_centerpoints(kps, Constants.right_hand_center_ref)
        kps = kps[:, Constants.body + Constants.face + Constants.left_hand + Constants.righ_hand+[-1, -2]]

        #generate gaussian heatmap
        kps = rearrange(kps, 'n k yx -> (n k) yx')
        # gmap = gaussian_heatmap(self.dw_pose_input_size[0], self.dw_pose_input_size[1], kps[:, 0], kps[:, 1], self.sigmas, input.device)
        gmap = gaussian_heatmap(
            valid_h, valid_w, kps[:, 0], kps[:, 1], self.sigmas, patch_height, patch_width, input.device)
        gmap = rearrange(gmap, '(n k) s h w -> n k s h w', n=N)
        gmap = torch.max(gmap, dim=1, keepdim=False)[0]
        # [n s h w]
        return gmap
        
    def _save_heatmap(self, heatmap, path):
        # n s h w
        heatmap = heatmap[0]
        import matplotlib.pyplot as plt
        import os
        for i in range(heatmap.shape[0]):
            file_name = f"heatmap_stage_{i}.jpg"
            fpath = os.path.join(path, file_name)
            h = heatmap[i].cpu().numpy()
            plt.imshow(h, cmap='hot', interpolation='nearest')
            plt.savefig(fpath)

if __name__ == '__main__':
    ckpt = '/root/projects/sign_language_transformer/resources/dwpose-l/dw-ll_ucoco.pth'
    cfg = '/root/projects/sign_language_transformer/resources/dwpose-l/rtmpose-l_8xb64-270e_coco-ubody-wholebody-256x192.py'

    loss = HeatMapFocalLoss(
        dw_pose_cfg=cfg,
        dw_pose_ckpt=ckpt,
        dw_pose_input_size=(256, 192),
        input_size=(224, 224),
        weights=[1.0, 1.0],
        sigmas=[2, 1.5, 1.2, 1, 0.8],
        stages=[0, 1, 2, 3, 4]
    ).to('cuda:1')
    
    image = '../resources/test_image.png'
    cfg = Config.fromfile(cfg)
    std = torch.tensor(cfg.model.data_preprocessor.std)
    mean = torch.tensor(cfg.model.data_preprocessor.mean)
    
    #prepare data
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

    image = Image.open('./resources/test_image.png')
    image = central_crop(image, 224, 224)
    print(f'data type: {np.array(image).dtype}')

    data = torch.from_numpy(np.array(image)).float()
    data = (data - std) / mean
    print(f'std: {data.std()}, mean: {data.mean()}')

    data = rearrange(data, ' h w c ->() c h w')
    data = repeat(data, ' n c h w -> n c t h w', t=100)
    data = data.to('cuda:1')
    
    attention = torch.randn(100, 1, 12, 12, 65, 64).to('cuda:1')
    attention = F.softmax(attention, dim=-1)
    attention = rearrange(attention, 't n s heads keys (h w) -> t n s heads keys h w', h=8, w=8)
    loss, heatmap = loss._loss_heatmap(attention, data)
    print(loss)

    def blend_images(base_img, overlay_img, alpha=0.5):
        """Blend the base image with the overlay iMage."""
        return (1 - alpha) * base_img + alpha * overlay_img
    origin = np.array(image) / 255.0
    h = heatmap[0, -1].cpu().numpy()
    import cv2
    h = cv2.resize(h, (224, 224))
    def apply_colormap(heatmap):
        """Apply a colormap to the heatmap and normalize."""
        cmap = plt.get_cmap('jet')
        heatmap_colored = cmap(heatmap)  # Apply colormap
        return heatmap_colored[..., :3]  # Discard alpha channel
    h = apply_colormap(h)
    blend_image = blend_images(origin, h, alpha=0.6)
    import matplotlib.pyplot as plt
    plt.imshow(blend_image)
    plt.savefig('outputs/blend_image.jpg')
