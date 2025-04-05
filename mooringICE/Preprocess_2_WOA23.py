import netCDF4 as nc
import numpy as np
from astropy.time import Time
from datetime import datetime, timedelta
import scipy.interpolate as itp


# ---------- read data ----------
moorData = np.load(r'mooringICE.npz')
lat_moor = moorData['latitude']
lon_moor = moorData['longitude']

t1 = nc.Dataset(r'L:\graduation proj\data\WOA23\woa23_B5C2_t12_04.nc')
lon_idx = np.argmin(np.abs(t1.variables['lon'][:] - lat_moor))
lat_idx = np.argmin(np.abs(t1.variables['lat'][:] - lon_moor))
lon = t1.variables['lon'][lon_idx]
lat = t1.variables['lat'][lat_idx]
depth = t1.variables['depth'][:]
nz = len(depth)
t = np.zeros((27, nz))
s = np.zeros((27, nz))
k = 0
for i in range(6, 13):
    path1 = r'L:\graduation proj\data\WOA23\woa23_B5C2_t{:02}_04.nc'.format(i)
    path2 = r'L:\graduation proj\data\WOA23\woa23_B5C2_s{:02}_04.nc'.format(i)
    temp = nc.Dataset(path1)
    t[k] = temp.variables['t_an'][0, :, lat_idx, lon_idx]
    salt = nc.Dataset(path2)
    s[k] = salt.variables['s_an'][0, :, lat_idx, lon_idx]
    k += 1
for i in range(1, 13):
    path1 = r'L:\graduation proj\data\WOA23\woa23_B5C2_t{:02}_04.nc'.format(i)
    path2 = r'L:\graduation proj\data\WOA23\woa23_B5C2_s{:02}_04.nc'.format(i)
    temp = nc.Dataset(path1)
    t[k] = temp.variables['t_an'][0, :, lat_idx, lon_idx]
    salt = nc.Dataset(path2)
    s[k] = salt.variables['s_an'][0, :, lat_idx, lon_idx]
    k += 1
for i in range(1, 9):
    path1 = r'L:\graduation proj\data\WOA23\woa23_B5C2_t{:02}_04.nc'.format(i)
    path2 = r'L:\graduation proj\data\WOA23\woa23_B5C2_s{:02}_04.nc'.format(i)
    temp = nc.Dataset(path1)
    t[k] = temp.variables['t_an'][0, :, lat_idx, lon_idx]
    salt = nc.Dataset(path2)
    s[k] = salt.variables['s_an'][0, :, lat_idx, lon_idx]
    k += 1
np.savez(r'WOA23_st.npz', t=t, s=s, z=depth, loc=[lat, lon])
# create woa23 date
# woa23Date = []
# current_date = datetime(2015, 6, 15)
# while current_date <= datetime(2017, 8, 15):
#     woa23Date.append(current_date)
#     month = current_date.month + 1 if current_date.month < 12 else 1
#     year = current_date.year if month > 1 else current_date.year + 1
#     current_date = datetime(year, month, 15)
# woa23Date = [(i - datetime(1950, 1, 1)).days for i in woa23Date]
# t_hourly = np.zeros((len(moorDate), len(depth)))
# for m in range(len(depth)):
#     itp_t = itp.interp1d(woa23Date, t[:, m])
#     t_hourly[:, m] = itp_t(moorDate)
# t_grid = np.zeros((len(moorDate), len(depthCurr)))
# for k in range(len(moorDate)):
#     itp_t = itp.interp1d(-depth, t_hourly[k, :], fill_value=np.nan, bounds_error=False)
#     t_grid[k, :] = itp_t(depthCurr)

print('c')
