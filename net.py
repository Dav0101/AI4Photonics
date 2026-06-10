import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import torcwa
import numpy as np
from matplotlib import pyplot as plt
import os
import loss as lf


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
        self.lamb0 = torch.tensor(1550.,dtype=geo_dtype,device=device)    # nm
        self.theta = 10.01*(np.pi/180)    # radian

        # material
        self.substrate_eps = 1.46**2
        #silicon_eps = Materials.aSiH.apply(lamb0)**2
        self.silicon_eps = 3.5**2

        # geometry
        self.L = [4531., 1000.]            # nm / nm
        torcwa.rcwa_geo.dtype = geo_dtype
        torcwa.rcwa_geo.device = device
        torcwa.rcwa_geo.Lx = self.L[0]
        torcwa.rcwa_geo.Ly = self.L[1]
        torcwa.rcwa_geo.nx = 256
        torcwa.rcwa_geo.ny = 128
        torcwa.rcwa_geo.grid()
        torcwa.rcwa_geo.edge_sharpness = 1000.

        # layers
        self.layer0_thickness = 300.

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
        self.register_buffer("initial_rho", nn.Parameter(torch.rand(1, 2 ** n, 2 ** m)))

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 1, kernel_size=3, stride=2, padding=1)
        )

        #self.threshold = nn.Parameter(torch.tensor(0.0))

    def forward(self, solver, tau=1.0):
        """logits = self.net(self.initial_rho)
        balanced_logits = (logits - logits.mean()) + self.threshold"""
        # squeeze removes the first dimension (channels=1)
        rho = gumbel_sigmoid(self.net(self.initial_rho), tau=tau, hard=True).squeeze(0)
        print(rho)
        sigmas = []

        for angle in self.angles.tolist():
            jones_matrix = solver.solve(rho, angle)
            singular_values = torch.linalg.svdvals(jones_matrix)
            sigmas.append(torch.min(singular_values))

        return rho.squeeze(0), torch.stack(sigmas)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNetRCWA(n=11, m=10, step=360)
    model = model.to(device)

    # if there is a good model already found, load it.
    checkpoint_path = 'best_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        best_loss = checkpoint['best_loss']
        lr = 0.0001
    else:
        best_loss = float('inf')
        lr = 0.001

    solver = rcwa_solver(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.9))
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=50)

    loss_plot = []

    epochs = 2000

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        current_tau = max(0.1, 3.0 * (0.998 ** epoch))

        r, v = model(solver, tau=current_tau)
        print(r.shape)
        print(v)

        loss = -lf.harmonic_mean(v)
        loss_plot.append(loss.item())
        loss.backward()
        # clipping the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.05)
        optimizer.step()
        #scheduler.step(loss.item())
        
        print(f"Epoch {epoch+1} - Loss: {loss:.4f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            checkpoint = {
                'epoch': epoch,
                'best_loss': best_loss,
                'model_state': model.state_dict()
            }
            torch.save(checkpoint, 'best_model.pth')

    # plot the loss
    plt.plot([i for i in range(epochs)], loss_plot)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("loss.png", dpi=300, bbox_inches="tight")
    plt.show()
    # save the model
    #torch.save(model.state_dict(), 'model.pth')