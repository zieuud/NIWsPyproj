import numpy as np
import matplotlib.pyplot as plt


temp_moor = np.load(r'MoorData\SENSOR_temp_interpolate.npy')
temp_woa = np.load(r'ReanaData\WOA23_temp_grid.npy')
adcp = np.load(r'MoorData\ADCP_uv.npz')
mtime = adcp['mtime_adcp']
depth = adcp['depth_adcp']

plt.figure(1, figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.pcolormesh(mtime, depth, temp_moor, cmap='Reds', vmin=0, vmax=20)
plt.colorbar()
plt.ylim([depth[-1], 0])
plt.subplot(2, 1, 2)
plt.pcolor(mtime, depth, temp_woa.T, cmap='Reds', vmin=0, vmax=20)
plt.colorbar()
plt.ylim([depth[-1], 0])
plt.savefig(r'figures\check_3_temp_moor_woa23.jpg', dpi=300)
plt.show()
print('c')
