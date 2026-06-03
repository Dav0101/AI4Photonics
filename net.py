import torch
import torch.nn as nn
from torch import Tensor


def gumbel_sigmoid(logits: Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5) -> Tensor:
    gumbels = (-torch.rand_like(
        logits, memory_format=torch.legacy_contiguous_format, device=logits.device
    ).exponential_().log())
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.sigmoid()

    if hard:
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft

    return ret


def rcwa(rho : Tensor, angle : Tensor) -> Tensor:
    # TODO Placeholder
    return torch.tensor([[1+2j, 3-1j], [4+0j, 5+2j]], dtype=torch.complex64, device=rho.device)


class ConvNetRCWA(nn.Module):
    def __init__(self, n, m, step):
        super().__init__()

        self.n = n
        self.m = m
        self.step = step

        self.register_buffer("angles", torch.arange(0, 360, step))
        self.register_buffer("initial_rho", torch.rand(1, 2 ** n, 2 ** m))

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self):
        rho = gumbel_sigmoid(self.net(self.initial_rho), hard=True)
        sigmas = []

        for angle in self.angles:
            jones_matrix = rcwa(rho, angle)
            singular_values = torch.linalg.svdvals(jones_matrix)
            sigmas.append(torch.min(singular_values))

        return rho.squeeze(0), torch.stack(sigmas)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNetRCWA(n=2, m=3, step=36)
    model = model.to(device)

    r, v = model()

    print(r.shape)
    print(v)