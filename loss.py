import torch
from torch import Tensor
import kornia.morphology as kr


def geometric_component(v: Tensor):
    v_smooth = kr.closing(kr.opening(v))
    return torch.mean(torch.abs(v-v_smooth))


def sample_mean(v: Tensor):
    return v.mean()


def geometric_mean(v: Tensor):
    return torch.exp(sample_mean(torch.log(v)))


def harmonic_mean(v: Tensor):
    return v.numel() / torch.sum(1.0 / v + 1e-6)