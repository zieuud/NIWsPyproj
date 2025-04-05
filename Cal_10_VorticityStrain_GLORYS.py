from datetime import datetime, timedelta
import netCDF4 as nc
import numpy as np
import scipy.interpolate as itp
from astropy.time import Time

glorys = nc.Dataset(r'K:\grad_proj\GLORY\cmems_mod_glo_phy_my_0.083deg_P1D-m_20150801-20160731_0-2000_36-37_-33--32.nc')
lat = glorys.variables['latitude'][:]
lon = glorys.variables['longitude'][:]
time = glorys.variables['time'][:]
depth = glorys.variables['depth'][:]
uo = glorys.variables['uo'][:]
vo = glorys.variables['vo'][:]  # dimensions: (time, depth, latitude, longitude)
lat_moor = 36.23
lon_moor = -32.75
moorData = np.load('ADCP_uv.npz')
moorDate = moorData['mtime']
moorDepth = moorData['depth']
# ----------calculate the strain and vorticity----------
R = 6371000
dx = R * np.radians(0.083)
dy = R * np.cos(np.radians(36.25)) * np.radians(0.083)
latIdx = 3  # 36.25
lonIdx = 3  # -32.74992
dudx = (uo[:, :, latIdx, lonIdx + 1] - uo[:, :, latIdx, lonIdx - 1]) / (2 * dx)
dudy = (uo[:, :, latIdx + 1, lonIdx] - uo[:, :, latIdx - 1, lonIdx]) / (2 * dy)
dvdx = (vo[:, :, latIdx, lonIdx + 1] - vo[:, :, latIdx, lonIdx - 1]) / (2 * dx)
dvdy = (vo[:, :, latIdx + 1, lonIdx] - vo[:, :, latIdx - 1, lonIdx]) / (2 * dy)
vorticity = np.squeeze(dvdx - dudy)
strain = np.squeeze(np.sqrt((dudx - dvdy) ** 2 + (dudy + dvdx) ** 2))
# ---------- normalize to moor depth ----------
strain_moor = np.zeros((366, len(moorDepth)))
vorticity_moor = np.zeros((366, len(moorDepth)))
for i in range(len(moorDepth)):
    idx = np.argmin(np.abs(depth + moorDepth[i]))
    strain_moor[:, i] = strain[:, idx]
    vorticity_moor[:, i] = vorticity[:, idx]
# ---------- interpolate on time ----------
glorysDate = time / 86400 + Time(datetime(1970, 1, 1)).jd - 1721424.5 + 366
itp_t = itp.interp1d(glorysDate, strain_moor, axis=0)
strain_moor_hourly = itp_t(moorDate)
itp_t = itp.interp1d(glorysDate, vorticity_moor, axis=0)
vorticity_moor_hourly = itp_t(moorDate)
# ---------- save data ----------
np.save(r'ReanaData\GLORYS_strain.npy', strain_moor_hourly)
np.save(r'ReanaData\GLORYS_vorticity.npy', vorticity_moor_hourly)
# for debugging
print('c')
