from scipy.io import loadmat
import numpy as np
from datetime import datetime, timedelta
import scipy.interpolate as itp


# data = loadmat('uv_ADCP.mat')
sensor = loadmat(r'L:\graduation proj\data\MOOR\RidgeMix_1mooring\Mooring_data\temperature_hourly.mat')
adcp = loadmat(r'L:\graduation proj\data\MOOR\RidgeMix_1mooring\Mooring_data\uv_ADCP.mat')
adcp_filter = loadmat(r'L:\graduation proj\data\MOOR\RidgeMix_1mooring\Mooring_data\uv_ADCP_filter_NI.mat')

depth_sensor = -np.squeeze(sensor['depth_sensor'])
mtime_sensor = np.squeeze(sensor['mtime_sensor_hourly'])
temp = np.squeeze(sensor['temp_sensor_hourly'])
# np.savez(r'MoorData\SENSOR_temp.npz', depth_sensor=depth_sensor, mtime_sensor=mtime_sensor, temp=temp)

depth_adcp = -np.squeeze(adcp['depth_ADCP']).astype(float)
mtime_adcp = np.squeeze(adcp['mtime_ADCP'])
u = np.squeeze(adcp['u_ADCP'])
v = np.squeeze(adcp['v_ADCP'])
# quality control
u[[40, 41, 42, 118, 119, 169, 170, 171, 172, 173, 174, 175], :] = np.nan
v[[40, 41, 42, 118, 119, 169, 170, 171, 172, 173, 174, 175], :] = np.nan
u[119:180, 2620:] = np.nan
v[119:180, 2620:] = np.nan
u[180:, 3960:] = np.nan
v[180:, 3960:] = np.nan
ke = 1 / 2 * 1025 * (u ** 2 + v ** 2)
# np.savez(r'MoorData\ADCP_uv.npz',
#          depth_adcp=depth_adcp, mtime_adcp=mtime_adcp, u=u, v=v, ke=ke)

depth_adcp = -np.squeeze(adcp_filter['depth_ADCP']).astype(float)
mtime_adcp = np.squeeze(adcp_filter['mtime_ADCP'])
u_ni = np.squeeze(adcp_filter['u_ADCP_filter_NI'])
v_ni = np.squeeze(adcp_filter['v_ADCP_filter_NI'])
ke_ni = 1 / 2 * 1025 * (u_ni ** 2 + v_ni ** 2)
# np.savez(r'MoorData\ADCP_uv_ni_byYu.npz',
#          depth_adcp=depth_adcp, mtime_adcp=mtime_adcp, u_ni=u_ni, v_ni=v_ni, ke_ni=ke_ni)

temp_interp = np.copy(u) * np.nan
for i in range(temp.shape[-1]):
    itp_z = itp.interp1d(depth_sensor, temp[:, i], fill_value=(temp[-1, i], temp[0, i]), bounds_error=False)
    temp_interp[:, i] = itp_z(depth_adcp)
temp_interp[:9, :] = np.nan  # cancel extrapolation
np.save(r'MoorData\SENSOR_temp_interpolate.npy', temp_interp)

date1 = [datetime(1, 1, 1) + timedelta(days=m - 366) for m in mtime_adcp]
date2 = [datetime(1, 1, 1) + timedelta(days=m - 366) for m in mtime_sensor]
print('c')
