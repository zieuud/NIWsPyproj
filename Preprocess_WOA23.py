import netCDF4 as nc
import numpy as np


t1 = nc.Dataset(r'K:\grad_proj\WOA23\woa23_B5C2_t12_04.nc')
# print(t1.variables.keys())
# print(t1.variables)
lon_idx = np.argmin(np.abs(t1.variables['lon'][:] - (-32.75)))
lat_idx = np.argmin(np.abs(t1.variables['lat'][:] - 36.23))
lon = t1.variables['lon'][lon_idx]
lat = t1.variables['lat'][lat_idx]
depth = t1.variables['depth'][:]
t_an = np.zeros((12, 57))
s_an = np.zeros((12, 57))
for i in range(1, 13):
    path1 = r'K:\grad_proj\WOA23\woa23_B5C2_t{:02}_04.nc'.format(i)
    path2 = r'K:\grad_proj\WOA23\woa23_B5C2_s{:02}_04.nc'.format(i)
    temp = nc.Dataset(path1)
    t_an[i-1] = temp.variables['t_an'][0, :, lat_idx, lon_idx]
    salt = nc.Dataset(path2)
    s_an[i-1] = salt.variables['s_an'][0, :, lat_idx, lon_idx]

np.savez(r'ReanaData\WOA23_st.npz', t=t_an, s=s_an, z=depth, loc=[lat, lon])


