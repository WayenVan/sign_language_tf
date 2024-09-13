import cv2
import numpy as np

from mmflow.models.builder import build_flow_estimator
from mmengine.config import Config

# Read an image
image = cv2.imread("path_to_image.jpg")

# Map the image to [0, 1]
image = image.astype(np.float32) / 255.0

# Initialize the FlowNet2SD model

cfg = Config.fromfile("resources/GMA/gma_8x2_50k_kitti2015_288x960.py").to_dict()
model = build_flow_estimator(cfg)
