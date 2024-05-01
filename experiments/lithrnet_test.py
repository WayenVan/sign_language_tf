import torch
import sys
sys.path.append('./src')
from csi_sign_language.modules.lithrnet.build import build_litehrnet
from csi_sign_language.modules.lithrnet.litehrnet import IterativeHeadDownSample


model = build_litehrnet('resources/litehrnet_18_coco_256x192_.pth')
header = IterativeHeadDownSample(model.stages_spec['num_channels'][-1], model.conv_cfg, model.norm_cfg)


input = torch.ones([168, 3, 256, 192]).cuda()
a = model(input)
b = header(a)
print(a)

