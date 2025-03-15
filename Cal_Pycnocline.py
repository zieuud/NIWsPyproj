import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta
from astropy.time import Time
import scipy.interpolate as itp

glorys = nc.Dataset(
    r'K:\grad_proj\GLORY\cmems_mod_glo_phy_my_0.083deg_P1D-m_20150801-20160731_0-2000_36-37_-33--32_st.nc')
lat = glorys.variables['latitude'][:]
lon = glorys.variables['longitude'][:]
time = glorys.variables['time'][:]
depth = glorys.variables['depth'][:]
temperature = glorys.variables['thetao'][:]

lat_moor = 36.23
lon_moor = -32.75
moorData = np.load('ADCP_uv.npz')
moorDate = moorData['mtime']
moorDepth = moorData['depth']

latIdx = 3  # 36.25
lonIdx = 3  # -32.74992
temperature = np.squeeze(temperature[:, :, 3, 3])
# ---------- interpolate on time ----------
glorysDate = time / 86400 + Time(datetime(1970, 1, 1)).jd - 1721424.5 + 366
temperature_hourly = np.zeros((len(moorDate), len(depth)))
for i in range(np.size(temperature, 1)):
    itp_t = itp.interp1d(glorysDate, np.squeeze(temperature[:, i]))
    temperature_hourly[:, i] = itp_t(moorDate)
# ---------- calculate the pycnocline ----------
moorPycnocline = []
for i in range(len(moorDate)):
    idx = np.argmin(abs(abs(temperature_hourly[i, :] - temperature_hourly[i, 0]) - 0.5))
    moorPycnocline.append(depth[idx])
# ---------- save data ----------
np.save(r'ReanaData\GLORYS_pycnocline.npy', moorPycnocline)
print('c')
