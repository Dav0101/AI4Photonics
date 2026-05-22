import numpy as np
import torch
import matplotlib.pyplot as plt

import torcwa

torch.backends.cuda.matmul.allow_tf32 = False
sim_dtype = torch.complex64
geo_dtype = torch.float32
#device = torch.device('cuda')
device = 'cpu'

lam_min = 400.
lam_max = 600.
lam_step = 2.
DL = int((lam_max-lam_min)/lam_step)
lam_arr = torch.linspace(lam_min, lam_max, DL, dtype=geo_dtype, device=device)

nx, ny, edge_sharpness = 256, 256, 128
L = [400., 400.] # the period coincides by chance with the first wavelength ... 
geom = torcwa.geometry(Lx=L[0], Ly=L[1], nx=nx, ny=ny, edge_sharpness=edge_sharpness, dtype=geo_dtype, device=device)

# define layer
R = 120.
layer_geom = geom.circle(R=R, Cx=L[0]/2.,Cy=L[1]/2.)
layer_thickness = 400.

substrate_eps = 1.5**2

order_N = 10
order = [order_N, order_N]

# define array with all orders
o = np.arange(-order_N, order_N+1)
all_orders = np.stack(np.meshgrid(o,o),-1).reshape(-1,2)

T_sum = torch.zeros_like(lam_arr)
R_sum = torch.zeros_like(lam_arr)

for lam_idx in range(len(lam_arr)):
  
   sim = torcwa.rcwa(freq=1.0/lam_arr[lam_idx], order=order, L=L, dtype=sim_dtype, device=device)
   
   sim.add_input_layer(eps=1.0)
   sim.add_output_layer(eps=substrate_eps)
   sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0) # normal incidence

   layer_eps = layer_geom*substrate_eps + (1.-layer_geom)
   sim.add_layer(thickness=layer_thickness, eps=layer_eps)
   
   sim.solve_global_smatrix()

   rpp = sim.S_parameters(orders=all_orders, direction='f', port='r', polarization='pp')
   tpp = sim.S_parameters(orders=all_orders, direction='f', port='t', polarization='pp')
   rss = sim.S_parameters(orders=all_orders, direction='f', port='r', polarization='ss')
   tss = sim.S_parameters(orders=all_orders, direction='f', port='t', polarization='ss')
   rps = sim.S_parameters(orders=all_orders, direction='f', port='r', polarization='ps')
   tps = sim.S_parameters(orders=all_orders, direction='f', port='t', polarization='ps')
   rsp = sim.S_parameters(orders=all_orders, direction='f', port='r', polarization='sp')
   tsp = sim.S_parameters(orders=all_orders, direction='f', port='t', polarization='sp')

   # Diffraction efficiency of input p-polarization
   T_sum[lam_idx] = (torch.sum(torch.abs(tpp)**2) + torch.sum(torch.abs(tsp)**2))
   R_sum[lam_idx] = (torch.sum(torch.abs(rpp)**2) + torch.sum(torch.abs(rsp)**2))
   # or Diffraction efficiency of input s-polarization
   T_sum[lam_idx] = (torch.sum(torch.abs(tss)**2) + torch.sum(torch.abs(tps)**2))
   R_sum[lam_idx] = (torch.sum(torch.abs(rss)**2) + torch.sum(torch.abs(rps)**2))
   # or Average of diffraction efficiency
   T_sum[lam_idx] = ((torch.sum(torch.abs(tpp)**2) + torch.sum(torch.abs(tsp)**2)) + (torch.sum(torch.abs(tss)**2) + torch.sum(torch.abs(tps)**2)))/2
   R_sum[lam_idx] = ((torch.sum(torch.abs(rpp)**2) + torch.sum(torch.abs(rsp)**2)) + (torch.sum(torch.abs(rss)**2) + torch.sum(torch.abs(rps)**2)))/2

eps_out, _ = sim.return_layer(layer_num=0, nx=geom.nx, ny=geom.ny)

fig, axarr = plt.subplots(1,2, figsize=(16,4))
eps_im = axarr[0].imshow(eps_out.cpu().real, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2])
fig.colorbar(eps_im, ax=axarr[0])
axarr[0].set(title=f'$\epsilon$ of layer #0 (FO: {order_N})',
             xlabel='x [nm]', ylabel='y [nm]')
axarr[1].plot(lam_arr.cpu(), R_sum.cpu(), label='R')  
axarr[1].plot(lam_arr.cpu(), T_sum.cpu(), label='T')
axarr[1].set(title=f'spectrum (FO: {order_N})',
             xlabel='$\lambda$ [nm]',
             ylabel='normalized(?) intensity')
axarr[1].legend()
plt.show()