import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta
import scipy.interpolate as itp

glorys = nc.Dataset(r'K:\grad_proj\GLORY\cmems_mod_glo_phy_my_0.083deg_P1D-m_20150801-20160731_0-2000_36-37_-33--32.nc')
lat = glorys.variables['latitude'][:]
lon = glorys.variables['longitude'][:]
time = glorys.variables['time'][:]
depth = glorys.variables['depth'][:]
uo = glorys.variables['uo'][:]
vo = glorys.variables['vo'][:]  # dimensions: (time, depth, latitude, longitude)
uo_surface = np.squeeze(uo[:, 0, :, :])
vo_surface = np.squeeze(vo[:, 0, :, :])
lat_moor = 36.23  # 36.25 index 3
lon_moor = -32.75  # -32.74992 index 3
moorData = np.load('ADCP_uv.npz')
moorDate = moorData['mtime']
# ----------calculate the vorticity----------
dx = 0.083 * 110000
dy = 0.083 * 110000
latIdx = 3  # 36.25
lonIdx = 3  # -32.74992
dvdx = (vo_surface[:, latIdx, lonIdx+1] - vo_surface[:, latIdx, lonIdx-1]) / (2 * dx)
dudy = (uo_surface[:, latIdx+1, lonIdx] - uo_surface[:, latIdx-1, lonIdx]) / (2 * dy)
vorticity_moor = np.squeeze(dvdx - dudy)
# ---------interpolate on time----------
dates = [datetime(2016, 8, 1) + timedelta(days=i) for i in
         range((datetime(2017, 7, 31) - datetime(2016, 7, 31)).days + 1)]
glorysDate = [date.toordinal() for date in dates]
itp_t = itp.interp1d(glorysDate, vorticity_moor)
vorticity_moor_hourly = itp_t(moorDate)
# ---------- save vorticity ----------
np.save(r'ReanaData\GLORYS_vorticity.npy', vorticity_moor_hourly)

print('c')
