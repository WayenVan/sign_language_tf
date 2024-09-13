import torch
from torch import Tensor
from einops import reduce


def t_mse(y: Tensor, tau, epsilon=1e-8):
    """
    @param y: [..., T, C]
    @param tau: float
    @param reduce: str the reduction method
    @return: [...]
    """
    T, C = y.shape[-1], y.shape[-2]

    y_t = y
    y_tp = torch.cat((y[..., 0:1, :], y[..., :-1, :]), dim=-2).detach()

    delta_t = torch.abs(torch.log(y_t + epsilon) - torch.log(y_tp + epsilon))
    delta_t_hat = torch.where(delta_t < tau, tau, delta_t)

    return torch.sum(delta_t_hat**2, dim=(-1, -2)) / (T * C)


if __name__ == "__main__":
    y = torch.randn(2, 4, 6, 100, 144)
    y = torch.nn.functional.softmax(y, dim=-1)
    tau = 0.1
    result = t_mse(y, tau)
    print(result.shape)
