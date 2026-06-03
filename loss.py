import torch
from torch import Tensor


def geometric_component(rho):
    # TODO Placeholder
    return torch.tensor(0)


def sample_mean(v: Tensor):
    return v.mean()


def geometric_mean(v: Tensor):
    return torch.exp(sample_mean(torch.log(v)))


def harmonic_mean(v: Tensor):
    return v.numel() / torch.sum(1.0 / v)


def rcwa_loss(rho, sigmas, alpha=1):
    return - harmonic_mean(sigmas) + alpha * geometric_component(rho)


if __name__ == '__main__':
    print(torch.exp(torch.tensor(1)))