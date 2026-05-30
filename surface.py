
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torcwa

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

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.sigmoid(self.conv3(x))
        
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
        self.L = [256., 128.]            # nm / nm
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

    def solve(self, rho, phi):
        order = [10,4]
        sim = torcwa.rcwa(freq=1/self.lamb0,order=order,L=self.L,dtype=self.sim_dtype,device=self.device)
        sim.add_input_layer(eps=self.substrate_eps)
        sim.set_incident_angle(inc_ang=self.theta,azi_ang=phi)
        layer0_eps = rho*self.silicon_eps + (1.-rho)
        sim.add_layer(thickness=self.layer0_thickness,eps=layer0_eps)
        sim.solve_global_smatrix()
        x = sim.S_parameters(orders=[1,0],direction='forward',port='transmission',polarization='ss',ref_order=[0,0])
        ss.append(x.cpu().numpy())
        x = sim.S_parameters(orders=[1,0],direction='forward',port='transmission',polarization='pp',ref_order=[0,0])
        pp.append(x.cpu().numpy())
        x = sim.S_parameters(orders=[1,0],direction='forward',port='transmission',polarization='sp',ref_order=[0,0])
        sp.append(x.cpu().numpy())
        x = sim.S_parameters(orders=[1,0],direction='forward',port='transmission',polarization='ps',ref_order=[0,0])
        ps.append(x.cpu().numpy())


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
    rho = nn.parameter.Parameter(initial_rho)

    epochs = 10

    # in every epoch, 360 different azimuthal angles are tested
    for epoch in range(epochs):
        model.train()
        singular_values = []

        optimizer.zero_grad()
        
        for phi_deg in range(360):
            # conversion to radians
            phi = float(phi_deg)*(np.pi/180)
            # moving the surface on the GPU (or CPU)
            rho = rho.to(device)
            
            # the net produce the surface that must be given to torcwa
            conv_rho = model(rho)
            
            # the torcwa based solver extracts the singular values
            singular_values.append(solver.solve(rho,phi))          # smallest of the two singular values

        loss = armonic(singular_values)
        
        loss.backward()
        optimizer.step()
            
        print(f"Epoca {epoch+1}/{epochs} - Loss: {loss:.4f}")