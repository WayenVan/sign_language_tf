from csi_sign_language.modules.x3d import X3d
import torch

x3d = X3d().cuda()
intput = torch.ones((1, 3, 13, 160, 160)).cuda()
output = x3d(input)