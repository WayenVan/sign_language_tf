import sys
sys.path.append('src')
from csi_sign_language.modules.pooling import NormPoolTemp
import torch

p = NormPoolTemp(3, 1)
a = torch.rand(10, 2, 64, 24, 24).requires_grad_()
b: torch.Tensor = p(a)
c = torch.sum(b, dim=[i for i in range(len(b.shape))])
c.backward()
print(b.grad)