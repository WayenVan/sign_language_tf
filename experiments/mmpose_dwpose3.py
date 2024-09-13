import torch
from torchvision.transforms import functional as F
from mmpose.apis.inference import init_model, init_default_scope
from matplotlib import pyplot as plt
from mmengine import Config
from mmpose.models.heads import RTMCCHead, SimCCHead
from mmpose.models import TopdownPoseEstimator
from mmpose.apis import Pose2DInferencer, inference_topdown
from mmdet.models.backbones import CSPNeXt
from mmpose.codecs.simcc_label import SimCCLabel
from mmpose.models.losses import KLDiscretLoss
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


def decode(x, y, simcc_split_ratio):
    x_locs = x.argmax(dim=-1)
    y_locs = y.argmax(dim=-1)
    locs = torch.stack([x_locs, y_locs], dim=-1).to(x.dtype)
    locs /= simcc_split_ratio
    return locs


# cfg = Config.fromfile(
#     "resources/RTMPose/rtmpose-l_8xb64-270e_coco-wholebody-256x192.py"
# )
# ckpt = "resources/RTMPose/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth"
cfg = Config.fromfile("resources/SimCC/simcc_res50_8xb64-210e_coco-256x192.py")
ckpt = "resources/SimCC/simcc_res50_8xb64-210e_coco-256x192-8e0f5b59_20220919.pth"
init_default_scope("mmpose")
model = init_model(cfg, ckpt, device="cuda:1")
mean = cfg.model.data_preprocessor.mean
std = cfg.model.data_preprocessor.std
W, H = cfg.codec.input_size


# prepare data
image = Image.open("./resources/test_image.png")
image = central_crop(image, 224, 224)
image = image.resize((H, W), Image.Resampling.BICUBIC)

image = F.to_tensor(image)
image = F.normalize(image, mean, std)
image = image.to("cuda:1")
image.shape

# run model
x, y = model(image.unsqueeze(0), None, "tensor")
locs = decode(x, y, cfg.model.head.simcc_head.split_ratio)
print(locs.shape)
