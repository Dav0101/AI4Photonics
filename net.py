import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import torcwa
import numpy as np
from matplotlib import pyplot as plt
import loss as lf
import time
import kornia.morphology as kr


def gumbel_sigmoid(logits: Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5) -> Tensor:
    gumbels = (-torch.rand_like(
        logits, memory_format=torch.legacy_contiguous_format, device=logits.device
    ).exponential_().log())
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.sigmoid()

    if hard:
        """indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0"""
        y_hard = (y_soft > threshold).to(logits.dtype)

        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft

    return ret


class rcwa_solver():
    def __init__(self, device):
        self.device=device
        # Simulation environment
        # light
        self.sim_dtype = torch.complex64
        geo_dtype = torch.float32
        #self.lamb0 = torch.tensor(1550.,dtype=geo_dtype,device=device)    # nm
        self.lamb0 = torch.tensor(1550.,dtype=geo_dtype,device=device)    # nm
        self.theta = 0.01*(np.pi/180)    # radian

        # material
        self.substrate_eps = 1.46**2
        #silicon_eps = Materials.aSiH.apply(lamb0)**2
        self.silicon_eps = 3.5**2

        # geometry
        self.L = [4531., 1000.]            # nm / nm
        #self.L = [1087., 525.]            # nm / nm

        # layers
        self.layer0_thickness = 300.

        # geometry for the plot
        torcwa.rcwa_geo.dtype = geo_dtype
        torcwa.rcwa_geo.device = device
        torcwa.rcwa_geo.Lx = self.L[0]
        torcwa.rcwa_geo.Ly = self.L[1]
        torcwa.rcwa_geo.nx = 2048
        torcwa.rcwa_geo.ny = 512
        torcwa.rcwa_geo.grid()
        torcwa.rcwa_geo.edge_sharpness = 1000.

    # this returns the scattering matrix for the chosen deflection and incident orders
    def solve(self, rho: Tensor, phi: float) -> Tensor:
        # radius for m and n respectively
        order = [10,4]

        sim = torcwa.rcwa(freq=1/self.lamb0,order=order,L=self.L,dtype=self.sim_dtype,device=self.device)
        sim.add_input_layer(eps=self.substrate_eps)
        sim.set_incident_angle(inc_ang=self.theta,azi_ang=phi)
        epsilon = rho*self.silicon_eps + (1.-rho)
        sim.add_layer(thickness=self.layer0_thickness,eps=epsilon)

        sim.solve_global_smatrix()
        S_ss = sim.S_parameters(orders=[1,0],direction='forward',port='transmission',polarization='ss',ref_order=[0,0])
        S_pp = sim.S_parameters(orders=[1,0],direction='forward',port='transmission',polarization='pp',ref_order=[0,0])
        S_sp = sim.S_parameters(orders=[1,0],direction='forward',port='transmission',polarization='sp',ref_order=[0,0])
        S_ps = sim.S_parameters(orders=[1,0],direction='forward',port='transmission',polarization='ps',ref_order=[0,0])

        row1 = torch.stack([S_ss, S_ps])
        row2 = torch.stack([S_sp, S_pp])
        jones_matrix = torch.stack([row1, row2])
        print(jones_matrix)

        return jones_matrix


class ConvNetRCWA(nn.Module):
    def __init__(self, n, m, step):
        super().__init__()

        self.register_buffer("angles", torch.arange(0, 360, step, dtype=torch.float32))
        self.register_buffer("initial_rho", nn.Parameter(torch.rand(1, 1, 2 ** n, 2 ** m)))

        self.net = nn.Sequential(
            # first halving, 1024 x 256, reasoning
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),

            # second halving, 512 x 128, reasoning
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),

            # reasoning
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same'),
            nn.InstanceNorm2d(128),
            nn.ReLU(),

            # first doubling, 1024 x 256, reasoning
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding='same'),
            nn.InstanceNorm2d(64),
            nn.ReLU(),

            # second doubling, 2048 x 512, reasoning
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding='same'),
            nn.InstanceNorm2d(32),
            nn.ReLU(),

            # last reasoning
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding='same')    # desired size: 2048 x 512 (2^11 x 2^9)
        )

    def forward(self, solver, tau=1.0):
        # squeeze removes the dimensions with 1 (channels=1, batch_size = 1)
        rho = gumbel_sigmoid(self.net(self.initial_rho), tau=tau, hard=True).squeeze()
        print(rho)
        sigmas = []

        for angle in self.angles.tolist():
            jones_matrix = solver.solve(rho, angle)
            singular_values = torch.linalg.svdvals(jones_matrix)
            sigmas.append(torch.min(singular_values))

        return rho, torch.stack(sigmas)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNetRCWA(n=11, m=9, step=10)
    model = model.to(device)
    
    epochs = 1200
    warmup = 700
    max_geometric_weight = 60.0

    solver = rcwa_solver(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003, betas=(0.5, 0.9))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    rcwa_loss_plot = []
    geometric_loss_plot = []
    loss_plot = []
    best_rcwa_loss = float('inf')
    start_total_time = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # parameter to balance the gumbel
        current_tau = max(0.1, 3.0 * (0.995 ** epoch))
        # parameter to balance the geometric component of the loss
        if epoch < warmup:
            geometric_weight = 0.0
        else:
            progress = (epoch - warmup) / (epochs - warmup)
            geometric_weight = max_geometric_weight*progress

        r, v = model(solver, tau=current_tau)
        print(r.shape)
        print(v)

        rcwa_loss = -lf.harmonic_mean(v)
        geometric_loss = lf.geometric_component(r)
        loss = rcwa_loss + geometric_weight*geometric_loss

        rcwa_loss_plot.append(rcwa_loss.item())
        geometric_loss_plot.append(geometric_loss.item())
        loss_plot.append(loss.item())

        loss.backward()
        # clipping the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.05)
        optimizer.step()
        # reduce the learning rate
        scheduler.step(loss.item())
        
        print(f"Epoch {epoch+1} - Loss: {loss:.4f}")

        if epoch > warmup and rcwa_loss.item() < best_rcwa_loss:
            best_rcwa_loss = rcwa_loss.item()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start_total_time
    print(f"total time: {total_time/60:.2f} minutes")
    print(best_rcwa_loss)

    # save the model
    torch.save(model.state_dict(), 'best_model.pth')

    # plot the loss
    plt.figure()
    plt.plot(range(epochs), rcwa_loss_plot, label="rcwa loss", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("rcwa_loss.png", dpi=300, bbox_inches="tight")
    plt.show()

    # plot the loss
    plt.figure()
    plt.plot(range(epochs), geometric_loss_plot, label="geometric loss", color ="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("geometric_loss.png", dpi=300, bbox_inches="tight")
    plt.show()

    # plot the loss
    plt.figure()
    plt.plot(range(epochs), loss_plot, label="total_loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.png", dpi=300, bbox_inches="tight")
    plt.show()

    # plot the surface
    plt.figure()
    x_axis = torcwa.rcwa_geo.x.cpu().numpy()
    y_axis = torcwa.rcwa_geo.y.cpu().numpy()
    plt.imshow(torch.transpose(r,-2,-1).detach().cpu(),origin='lower',extent=[x_axis[0],x_axis[-1],y_axis[0],y_axis[-1]])
    plt.xlim([0,torcwa.rcwa_geo.Lx])
    plt.ylim([0,torcwa.rcwa_geo.Ly])
    plt.colorbar()
    plt.savefig("rho.png", dpi=300, bbox_inches="tight")
    plt.show()

    # plot the fixed surface
    plt.figure()
    x_axis = torcwa.rcwa_geo.x.cpu().numpy()
    y_axis = torcwa.rcwa_geo.y.cpu().numpy()
    kernel = torch.ones(5, 5, device=v.device, dtype=v.dtype)
    plt.imshow(
        torch.transpose(
            kr.closing(kr.opening(r.unsqueeze(0).unsqueeze(0), kernel), kernel),-2,-1
        ).detach().cpu(),origin='lower',extent=[x_axis[0],x_axis[-1],y_axis[0],y_axis[-1]]
    )
    plt.xlim([0,torcwa.rcwa_geo.Lx])
    plt.ylim([0,torcwa.rcwa_geo.Ly])
    plt.colorbar()
    plt.savefig("rho.png", dpi=300, bbox_inches="tight")
    plt.show()