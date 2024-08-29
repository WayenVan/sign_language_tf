import torch


def t_mse(y, tau, epsilon=1e-8):
    """
    @param y: [T, B, C]
    """
    T, _, C = y.shape

    y_t = y
    y_tp = torch.cat((y[0:1], y[:-1]), dim=0).detach()

    delta_t = torch.abs(torch.log(y_t + epsilon) - torch.log(y_tp + epsilon))
    delta_t_hat = torch.where(delta_t < tau, tau, delta_t)

    return torch.sum(delta_t_hat**2, dim=(0, 2)) / (T * C)


if __name__ == "__main__":
    y = torch.randn(10, 16, 2)
    y = torch.nn.functional.softmax(y, dim=-1)
    tau = 0.1
    result = t_mse(y, tau)
    print(result)
