import xarray as xr
import numpy as np


ds = xr.open_dataset(r'L:\graduation proj\adaptor.mars.internal-1665023026.1927767-912-15-f9b76ec1-0235-44bf-9aab-cf4c219f672b.grib', engine='cfgrib')
# print(ds)
time = ds['time'].values
lon = ds['longitude'].values
lat = ds['latitude'].values
u10 = ds['u10'].values  # u10(time, latitude, longitude) float32
v10 = ds['v10'].values

lon_idx = np.argmin(np.abs(lon - (-32.75)))
lat_idx = np.argmin(np.abs(lat - 36.23))
time_start = 6461
time_end = 13111
time_adcp = time[6461:13111]
u10Moor = u10[time_start:time_end, lat_idx, lon_idx]
v10Moor = v10[time_start:time_end, lat_idx, lon_idx]
# np.savez(r'ReanaData/ERA5_wind_moor.npz', u10=u10Moor, v10=v10Moor)
print('c')
