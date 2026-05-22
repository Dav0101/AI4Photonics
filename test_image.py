'''
TORCWA Example6
Topology optimization - Maximize 1st order diffraction

'''
# Import
import numpy as np
import torch
from torchvision import transforms
import scipy.io
from matplotlib import pyplot as plt
import time
from PIL import Image

import torcwa
import Materials

# Hardware
# If GPU support TF32 tensor core, the matmul operation is faster than FP32 but with less precision.
# If you need accurate operation, you have to disable the flag below.
torch.backends.cuda.matmul.allow_tf32 = False
sim_dtype = torch.complex64
geo_dtype = torch.float32
#device = torch.device('cuda')
device = 'cpu'

# Simulation environment
# light
lamb0 = torch.tensor(600.,dtype=geo_dtype,device=device)    # nm
theta = 0.01*(np.pi/180)    # radian
phi = 0.*(np.pi/180)    # radian

# material
substrate_eps = 1.46**2
#silicon_eps = Materials.aSiH.apply(lamb0)**2
silicon_eps = 3.5**2

# geometry
L = [1087., 525.]            # nm / nm
torcwa.rcwa_geo.dtype = geo_dtype
torcwa.rcwa_geo.device = device
torcwa.rcwa_geo.Lx = L[0]
torcwa.rcwa_geo.Ly = L[1]
torcwa.rcwa_geo.nx = 404
torcwa.rcwa_geo.ny = 128
torcwa.rcwa_geo.grid()
torcwa.rcwa_geo.edge_sharpness = 1000.

x_axis = torcwa.rcwa_geo.x.cpu()
y_axis = torcwa.rcwa_geo.y.cpu()

# layers
layer0_thickness = 300.

img = scipy.io.loadmat("rho_Fan.mat")
rho = torch.as_tensor(img['rho_Fan'])

"""plt.imshow(torch.transpose(rho,-2,-1).cpu(),origin='lower',extent=[x_axis[0],x_axis[-1],y_axis[0],y_axis[-1]])
plt.title('Layer 0')
plt.xlim([0,L[0]])
plt.xlabel('x (nm)')
plt.ylim([0,L[1]])
plt.ylabel('y (nm)')
plt.colorbar()
plt.show()"""

data={}

for k in range(600,1050,50):
    lamb0 = torch.tensor(float(k),dtype=geo_dtype,device=device)    # nm
    order = [10,4]
    sim = torcwa.rcwa(freq=1/lamb0,order=order,L=L,dtype=sim_dtype,device=device)
    sim.add_input_layer(eps=substrate_eps)
    sim.set_incident_angle(inc_ang=theta,azi_ang=phi)
    layer0_eps = rho*silicon_eps + (1.-rho)
    sim.add_layer(thickness=layer0_thickness,eps=layer0_eps)
    sim.solve_global_smatrix()
    t1ss = sim.S_parameters(orders=[1,0],direction='forward',port='transmission',polarization='ss',ref_order=[0,0])
    t1pp = sim.S_parameters(orders=[1,0],direction='forward',port='transmission',polarization='pp',ref_order=[0,0])
    t1sp = sim.S_parameters(orders=[1,0],direction='forward',port='transmission',polarization='sp',ref_order=[0,0])
    t1ps = sim.S_parameters(orders=[1,0],direction='forward',port='transmission',polarization='ps',ref_order=[0,0])
    data[str(lamb0)] = {'tss':t1ss.cpu().numpy(),'tpp':t1pp.cpu().numpy(), 'tsp':t1sp.cpu().numpy(),'tps':t1ps.cpu().numpy()}

scipy.io.savemat(f'test.mat',data)