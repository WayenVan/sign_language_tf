import torch
from einops import rearrange
import cv2
import sys
sys.path.append('src')
from csi_sign_language.modules.flownet2.models import FlowNet2SD
from PIL import Image
import numpy as np
from csi_sign_language.modules.flownet2.utils.flow_utils import flow2img
from matplotlib import pyplot as plt
from csi_sign_language.modules.slr_base.x3d_encoder import TemporalShift
from omegaconf import OmegaConf
from hydra.utils import instantiate
import sys
import os
os.chdir('/home/jingyan/Documents/sign_language_rgb')


checkpoint = torch.load('resources/FlowNet2-SD_checkpoint.pth.tar')
flownet2sd = FlowNet2SD(color_range=[-1., 1.]).cuda()
flownet2sd.load_state_dict(checkpoint['state_dict'], strict=True)



# frame_t = Image.open('resources/1.jpg')
# frame_tp = Image.open('resources/0.jpg')
# images = [frame_tp, frame_t]
# images = np.array(images).transpose(3, 0, 1, 2)
# im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()
# im = (im/255.) *  2.0 - 1

config = OmegaConf.load('configs/train/x3d_flownet_trans.yaml')
loader = instantiate(config.data.train_loader)
data = next(iter(loader))
im = data['video']
t_length = data['video_length'].cuda()
im = TemporalShift()(im, t_length)
frame_tp = 255 *(im[0, 0, :, 2].cpu().numpy() + 1) / 2
frame_t = 255 * (im[1, 0, :, 2].cpu().numpy() + 1) / 2

frame_t =  frame_t.astype('uint8').transpose(1, 2, 0)
frame_tp =  frame_tp.astype('uint8').transpose(1, 2, 0)
frame_t = cv2.cvtColor(frame_t, cv2.COLOR_RGB2GRAY)
frame_tp = cv2.cvtColor(frame_tp, cv2.COLOR_RGB2GRAY)

im = rearrange(im, 'f n c t h w -> (n t) c f h w')


flownet2sd.eval().cuda()
sd_output = flownet2sd(im)[0][0]

flow_sd = sd_output.data.cpu().numpy().transpose(1, 2, 0)
flow_sd_img = flow2img(flow_sd)

# Image.fromarray(flow_c_img).save('resources/flow_c.png')
# Image.fromarray(flow_sd_img).save('resources/flow_sd.png')

fig, axs = plt.subplots(1, 2)
axs[0].imshow(frame_t, cmap='gray')
axs[0].set_title('frame_t')
axs[1].imshow(frame_tp, cmap='gray')
axs[1].set_title('frame_tp')
fig.savefig('resources/frames.png', bbox_inches='tight') 

fig, axs = plt.subplots(1, 2)
axs[0].set_title('FlowNet2C')
axs[1].imshow(flow_sd_img)
axs[1].set_title('FlowNet2SD')
fig.savefig('resources/flow.png', bbox_inches='tight') 



