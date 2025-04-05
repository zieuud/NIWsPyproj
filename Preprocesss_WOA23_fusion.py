import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta
import scipy.interpolate as itp


# read sensor data
sensor = np.load(r'MoorData/SENSOR_temp.npz')
timeSensor = sensor['mtime_sensor']
depthSensor = sensor['depth_sensor']
tempSensor = sensor['temp']
date = [datetime(1, 1, 1) + timedelta(days=m - 366) for m in timeSensor]  # from 2015-09-28 00:00:03 to 2016-07-01 06:00:00
nt = len(timeSensor)

# read woa23 temperature salinity monthly
woa23Monthly = nc.Dataset(r'L:/graduation proj/data/WOA23/woa23_B5C2_t12_04.nc')
lonIdx = np.argmin(np.abs(woa23Monthly.variables['lon'][:] - (-32.75)))
latIdx = np.argmin(np.abs(woa23Monthly.variables['lat'][:] - 36.23))
depthMonthly = -np.array(woa23Monthly.variables['depth'][:])
nzMonthly = len(depthMonthly)
tempMonthly = np.zeros((12, nzMonthly))
saltMonthly = np.zeros((12, nzMonthly))
for i in range(1, 13):
    path1 = r'L:\graduation proj\data\WOA23\woa23_B5C2_t{:02}_04.nc'.format(i)
    path2 = r'L:\graduation proj\data\WOA23\woa23_B5C2_s{:02}_04.nc'.format(i)
    temp = nc.Dataset(path1)
    tempMonthly[i-1, :] = np.array(temp.variables['t_an'][0, :, latIdx, lonIdx])
    salt = nc.Dataset(path2)
    saltMonthly[i-1, :] = np.array(salt.variables['s_an'][0, :, latIdx, lonIdx])

# read woa23 salinity seasonal
woa23Seasonal = nc.Dataset(r'L:/graduation proj/data/WOA23/woa23_B5C2_s13_04.nc')
lonIdx = np.argmin(np.abs(woa23Seasonal.variables['lon'][:] - (-32.75)))
latIdx = np.argmin(np.abs(woa23Seasonal.variables['lat'][:] - 36.23))
depthSeasonal = -np.array(woa23Seasonal.variables['depth'][:])
nzSeasonal = len(depthSeasonal)
saltSeasonal = np.zeros((4, nzSeasonal))
for i in range(13, 17):
    path = r'L:\graduation proj\data\WOA23\woa23_B5C2_s{:02}_04.nc'.format(i)
    salt = nc.Dataset(path)
    saltSeasonal[i-13, :] = np.array(salt.variables['s_an'][0, :, latIdx, lonIdx])

# temperature fusion
depthFusion = np.concatenate([depthMonthly[:19], depthSensor])
nzFusion = len(depthFusion)
tempFusion = np.zeros((nt, nzFusion))
for i in range(nt):
    m = date[i].month
    tempFusion[i, :19] = tempMonthly[m-1, :19]
    tempFusion[i, 19:] = tempSensor[i, :]

# salinity fusion
nz2 = np.argwhere(depthSeasonal == -2300)[0][0] + 1
saltFusion = np.zeros((nt, nzFusion))
for i in range(nt):
    m = date[i].month
    if m == 1 or 2 or 3:
        s = 1
    elif m == 3 or 5 or 6:
        s = 2
    elif m == 7 or 8 or 9:
        s = 3
    else:
        s = 4
    salt = np.concatenate([saltMonthly[m - 1, :], saltSeasonal[s - 1, 57:nz2]])
    itp_t = itp.interp1d(depthSeasonal[:nz2], salt)
    saltFusion[i, :] = itp_t(depthFusion)

np.savez(r'ReanaData/WOA23_st_fusion.npz', tempFusion=tempFusion, saltFusion=saltFusion, depthFusion=depthFusion)

print('c')
