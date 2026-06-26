import torch
from torch import Tensor
import kornia.morphology as kr


def geometric_component(v: Tensor):
    v_4d = v.unsqueeze(0).unsqueeze(0)
    kernel = torch.ones(5, 5, device=v.device, dtype=v.dtype)

    v_smooth_4d = kr.closing(kr.opening(v_4d, kernel), kernel)
    v_smooth = v_smooth_4d.squeeze()

    return torch.mean(torch.abs(v-v_smooth))


def sample_mean(v: Tensor):
    return v.mean()


def geometric_mean(v: Tensor):
    return torch.exp(sample_mean(torch.log(v)))


def harmonic_mean(v: Tensor):
    return v.numel() / torch.sum(1.0 / v + 1e-6)

"""
access email: metamaterial.nest@gmail.com
pw: gigaflop

phone number: 3382637412

dugong pin: 636953

pw user: ai4photonics

inc ang for second surface 13.54"""