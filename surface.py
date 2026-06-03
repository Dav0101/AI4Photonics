
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torcwa
from matplotlib import pyplot as plt

class MSshaper(nn.Module):
    def __init__(self):
        super(MSshaper, self).__init__()

        # three conv layers
        # each layer must halve the tensor size
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=1
        )
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.5)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.5)
        nn.init.normal_(self.conv3.weight, mean=0.0, std=0.5)

        self.beta = 100.0

    def forward(self, x):
        print(x)
        x = F.leaky_relu(self.conv1(x))
        print(x)
        x = F.leaky_relu(self.conv2(x))
        print(x)
        x = self.conv3(x)
        x = x - x.mean()
        x = F.sigmoid(x * self.beta)
        print(x)
        
        return x
    
class rcwa_solver():
    def __init__(self, device):
        self.device=device
        # Simulation environment
        # light
        self.sim_dtype = torch.complex64
        geo_dtype = torch.float32
        self.lamb0 = torch.tensor(600.,dtype=geo_dtype,device=device)    # nm
        self.theta = 10.01*(np.pi/180)    # radian

        # material
        self.substrate_eps = 1.46**2
        #silicon_eps = Materials.aSiH.apply(lamb0)**2
        self.silicon_eps = 3.5**2

        # geometry
        self.L = [2048., 1024.]            # nm / nm
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
    def solve(self, rho, phi):
        # radius for m and n respectively
        order = [10,4]

        sim = torcwa.rcwa(freq=1/self.lamb0,order=order,L=self.L,dtype=self.sim_dtype,device=self.device)
        sim.add_input_layer(eps=self.substrate_eps)
        sim.set_incident_angle(inc_ang=self.theta,azi_ang=phi)
        epsilon = rho*self.silicon_eps + (1.-rho)
        sim.add_layer(thickness=self.layer0_thickness,eps=epsilon)

        sim.solve_global_smatrix()
        S_ss = sim.S_parameters(orders=[1,0],direction='forward',port='transmission',polarization='ss',ref_order=[0,0])
        print(S_ss)
        S_pp = sim.S_parameters(orders=[1,0],direction='forward',port='transmission',polarization='pp',ref_order=[0,0])
        S_sp = sim.S_parameters(orders=[1,0],direction='forward',port='transmission',polarization='sp',ref_order=[0,0])
        S_ps = sim.S_parameters(orders=[1,0],direction='forward',port='transmission',polarization='ps',ref_order=[0,0])

        row1 = torch.stack([S_ss, S_ps])
        row2 = torch.stack([S_sp, S_pp])
        return torch.stack([row1, row2])
    
def armonic_mean(x):
    return 1.0 / torch.mean(1.0 / x)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # If GPU supports TF32 tensor core, the matmul operation is faster than FP32 but with less precision.
    # If you need accurate operation, you have to disable the flag below.
    torch.backends.cuda.matmul.allow_tf32 = False

    model = MSshaper()
    solver = rcwa_solver(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    # rho: binary tensor of size 2048 x 1024
    initial_rho = torch.randint(low=0, high=2, size=(2048,1024))
    rho = nn.parameter.Parameter(initial_rho.float())

    epochs = 20

    # in every epoch, 360 different azimuthal angles are tested
    for epoch in range(epochs):
        model.train()
        min_singular_values = []

        optimizer.zero_grad()

        # moving the surface on the GPU (or CPU)
        rho_dev = rho.to(device)
        
        # the net produce the surface that must be given to torcwa
        # it's necessary to add the dimensions 1 for channels and for batch
        # when we use conv2d
        rho_input = rho_dev.unsqueeze(0).unsqueeze(0)
        conv_rho = model(rho_input).squeeze(0).squeeze(0)
        
        for phi_deg in range(0,360,20):
            # conversion to radians
            phi = float(phi_deg)*(np.pi/180)

            # torcwa solver extracts the 2x2 scattering tensor
            S_matrix = solver.solve(conv_rho,phi)

            # singular values of the tensor
            singular_values = torch.linalg.svdvals(S_matrix)
            
            # the torcwa based solver extracts the singular values
            min_singular_values.append(singular_values[-1])          # smallest of the two singular values

        # conver the singular values in a single 1D tensor
        min_singular_values = torch.stack(min_singular_values)
        loss = -armonic_mean(min_singular_values)
        
        loss.backward()
        optimizer.step()
            
        print(f"Epoca {epoch+1}/{epochs} - Loss: {loss:.4f}")

    # plotting the suface
    x_axis = torcwa.rcwa_geo.x.cpu()
    y_axis = torcwa.rcwa_geo.y.cpu()
    plt.imshow(torch.transpose(rho,-2,-1).detach().cpu(),origin='lower',extent=[x_axis[0],x_axis[-1],y_axis[0],y_axis[-1]])
    plt.title('Layer 0')
    plt.xlim([0,torcwa.rcwa_geo.Lx])
    plt.xlabel('x (nm)')
    plt.ylim([0,torcwa.rcwa_geo.Ly])
    plt.ylabel('y (nm)')
    plt.colorbar()
    plt.show()

    # save the model
    torch.save(model.state_dict(), 'model.pth')
    torch.save(rho.data, 'rho.pt')