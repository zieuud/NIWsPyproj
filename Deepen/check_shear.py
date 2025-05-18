import matplotlib.pyplot as plt
import numpy as np


uv_ni = np.load(r'../MoorData/ADCP_uv_ni.npz')
u_ni = uv_ni['u_ni']
v_ni = uv_ni['v_ni']
ke_ni = uv_ni['KE_ni']
moor = np.load(r'../MoorData/ADCP_uv.npz')
depth = moor['depth_adcp']
time = moor['mtime_adcp']
dz = 8

shear = np.sqrt((u_ni[:, 1:] - u_ni[:, :-1]) ** 2 + (v_ni[:, 1:] - v_ni[:, :-1]) ** 2)

plt.figure(1, figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.pcolormesh(time, depth[:-1], shear.T, vmin=0, vmax=5e-2)
plt.colorbar()
plt.subplot(2, 1, 2)
plt.pcolormesh(time, depth[:], ke_ni.T, vmin=0, vmax=10)
plt.colorbar()
plt.show()
