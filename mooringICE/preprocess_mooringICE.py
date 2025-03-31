import matplotlib.pyplot as plt
import netCDF4 as nc
from datetime import datetime, timedelta
import numpy as np
import scipy.interpolate as itp

iceCurr = nc.Dataset(r'L:\graduation proj\data\MOOR\Data\RREX_ICE_CURR_hourly.nc')
# print(iceCurr.variables)
latitude = np.array(iceCurr.variables['LATITUDE'][:])
longitude = np.array(iceCurr.variables['LONGITUDE'][:])
mtime = np.array(iceCurr.variables['TIME'][:] ) # days since 1950-01-01 00:00:00
time = np.tile(mtime, (29, 1)).T
depthCurr = -np.array(iceCurr.variables['DEPTH'][:])
u = np.array(iceCurr.variables['UCUR'][:])
v = np.array(iceCurr.variables['VCUR'][:])
# quality control
u[np.abs(u) > 10] = np.nan
v[np.abs(v) > 10] = np.nan
ke = 1 / 2 * 1025 * (u ** 2 + v ** 2)

iceTemp = nc.Dataset(r'L:\graduation proj\data\MOOR\Data\RREX_ICE_TS_hourly.nc')
# print(iceTemp.variables)
depthTemp = -np.array(iceTemp.variables['DEPTH'][:])
temp = np.array(iceTemp.variables['TEMP'][:])

# np.savez(r'mooringICE.npz',
#          latitude=latitude, longitude=longitude, depthCurr=depthCurr[0, :], mtime=time,
#          u=u, v=v, ke=ke, depthTemp=depthTemp[0, :], temp=temp)

temp_interp = np.copy(u) * np.nan
for i in range(temp.shape[0]):
    itp_z = itp.interp1d(depthTemp[i, :], temp[i, :], fill_value=(temp[i, -1], temp[i, 0]), bounds_error=False)
    temp_interp[i, :] = itp_z(depthCurr[i, :])

date = [datetime(1950, 1, 1) + timedelta(days=m) for m in mtime]

# plt.figure()
# plt.pcolor(time, depthCurr, u, cmap='coolwarm', vmin=-0.4, vmax=0.4)
# plt.colorbar()
# plt.show()
print('c')
