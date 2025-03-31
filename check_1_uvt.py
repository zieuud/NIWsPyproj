import numpy as np
import matplotlib.pyplot as plt


adcp = np.load(r'MoorData\ADCP_uv.npz')
depth_adcp = adcp['depth_adcp']
mtime = adcp['mtime_adcp']
u = adcp['u']
v = adcp['v']
ke = adcp['ke']
sensor = np.load(r'MoorData\SENSOR_temp.npz')
depth_sensor = sensor['depth_sensor']
temp = sensor['temp']
uv_ni = np.load(r'MoorData\ADCP_uv_ni_byYu.npz')
u_ni = uv_ni['u_ni']
v_ni = uv_ni['v_ni']
ke_ni = uv_ni['ke_ni']

t1, h1 = np.meshgrid(mtime, depth_adcp)
t2, h2 = np.meshgrid(mtime, depth_sensor)

plt.figure(2, figsize=(10, 8))
plt.subplot(4, 1, 1)
plt.pcolormesh(mtime, depth_adcp, u, cmap='coolwarm')
plt.colorbar()
plt.subplot(4, 1, 2)
plt.pcolormesh(mtime, depth_adcp, v, cmap='coolwarm')
plt.colorbar()
plt.subplot(4, 1, 3)
plt.pcolormesh(mtime, depth_adcp, ke, cmap='Oranges')
plt.colorbar()
plt.subplot(4, 1, 4)
plt.pcolormesh(mtime, depth_sensor, temp, cmap='Reds')
plt.colorbar()
plt.savefig(r'figures\check_1_uvt.jpg', dpi=300)
# plt.show()

plt.figure(3, figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.pcolormesh(mtime, depth_adcp, u_ni, cmap='coolwarm', vmin=-0.1, vmax=0.1)
plt.colorbar()
plt.subplot(3, 1, 2)
plt.pcolormesh(mtime, depth_adcp, v_ni, cmap='coolwarm', vmin=-0.1, vmax=0.1)
plt.colorbar()
plt.subplot(3, 1, 3)
plt.pcolormesh(mtime, depth_adcp, ke_ni, cmap='Oranges', vmin=0, vmax=10)
plt.colorbar()
plt.savefig(r'figures\check_1_uv_filter_ByYu.jpg', dpi=300)
plt.show()
print('c')
