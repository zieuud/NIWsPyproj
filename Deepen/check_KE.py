import matplotlib.pyplot as plt
import numpy as np

uv_ni_yu = np.load(r'../MoorData/ADCP_uv_ni_byYu.npz')
uv_ni = np.load(r'../MoorData/ADCP_uv_ni.npz')
u_ni1 = uv_ni_yu['u_ni']
v_ni1 = uv_ni_yu['v_ni']
u_ni2 = uv_ni['u_ni']
v_ni2 = uv_ni['v_ni']
moor = np.load(r'../MoorData/ADCP_uv.npz')
depth = moor['depth_adcp']
time = moor['mtime_adcp']
ke_ni1 = 1 / 2 * 1025 * (u_ni1 ** 2 + v_ni1 ** 2)
ke_ni2 = 1 / 2 * 1025 * (u_ni2 ** 2 + v_ni2 ** 2)


plt.figure(1, figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.pcolormesh(time, depth, ke_ni1.T, cmap='Reds', vmin=0, vmax=20, shading='auto')
plt.colorbar()
plt.subplot(2, 1, 2)
plt.pcolormesh(time, depth, ke_ni2.T, cmap='Reds', vmin=0, vmax=20, shading='auto')
plt.colorbar()

plt.show()
