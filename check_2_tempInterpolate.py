import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as itp


def interp_on_depth(var, depth1, depth2):
    itp_t = itp.interp1d(depth1, var, fill_value=(var[-1], var[0]), bounds_error=False)
    return itp_t(depth2)


sensor = np.load(r'MoorData\SENSOR_temp.npz')
mtime = sensor['mtime_sensor']
depth_sensor = sensor['depth_sensor']
temp = sensor['temp']

adcp = np.load(r'MoorData\ADCP_uv.npz')
depth_adcp = adcp['depth_adcp']
u = adcp['u']

# temp_interp = np.copy(u) * np.nan
# for i in range(temp.shape[-1]):
#     temp_interp[:, i] = interp_on_depth(temp[:, i], depth_sensor, depth_adcp)
# temp_interp[:9, :] = np.nan
temp_interp = np.load(r'MoorData\SENSOR_temp_interpolate.npy')

plt.figure(1, figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.pcolormesh(mtime, depth_sensor, temp, cmap='Reds')
plt.colorbar()
plt.ylim([depth_adcp[-1], 0])
plt.subplot(2, 1, 2)
plt.pcolormesh(mtime, depth_adcp, temp_interp, cmap='Reds')
plt.colorbar()
plt.ylim([depth_adcp[-1], 0])
# plt.savefig(r'figures\check_2_tempInterpolate.jpg', dpi=300)
plt.show()
print('c')
