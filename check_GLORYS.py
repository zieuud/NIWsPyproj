import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from astropy.time import Time
from datetime import datetime, timedelta
import scipy.interpolate as itp


def interp_on_time(var, date1, date2):
    itp_t = itp.interp1d(date1, var)
    return itp_t(date2)


glorys = nc.Dataset(r'K:\grad_proj\GLORY\cmems_mod_glo_phy_my_0.083deg_P1D-m_20150801-20160731_0-2000_36-37_-33--32.nc')
lat = glorys.variables['latitude'][:]
lon = glorys.variables['longitude'][:]
time = glorys.variables['time'][:]
glorysDate = time / 86400 + Time(datetime(1970, 1, 1)).jd - 1721424.5 + 366
depth = glorys.variables['depth'][:]
uo = glorys.variables['uo'][:]
vo = glorys.variables['vo'][:]
lat_moor = 36.23
lon_moor = -32.75
moorData = np.load('ADCP_uv.npz')
moorDate = moorData['mtime']
moorDepth = moorData['depth']
dateForPlot = [datetime(1, 1, 1) + timedelta(days=m-366) for m in moorDate]
# ---------- calculate the vorticity and divergence ----------
R = 6371000
dx = R * np.radians(0.083)
dy = R * np.cos(np.radians(36.25)) * np.radians(0.083)
latIdx = 3  # 36.25
lonIdx = 3  # -32.74992
uMoor = interp_on_time(np.squeeze(uo[:, 0, latIdx, lonIdx]), glorysDate, moorDate)
vMoor = interp_on_time(np.squeeze(vo[:, 0, latIdx, lonIdx]), glorysDate, moorDate)
dudx = (uo[:, :, latIdx, lonIdx + 1] - uo[:, :, latIdx, lonIdx - 1]) / (2 * dx)
dudy = (uo[:, :, latIdx + 1, lonIdx] - uo[:, :, latIdx - 1, lonIdx]) / (2 * dy)
dvdx = (vo[:, :, latIdx, lonIdx + 1] - vo[:, :, latIdx, lonIdx - 1]) / (2 * dx)
dvdy = (vo[:, :, latIdx + 1, lonIdx] - vo[:, :, latIdx - 1, lonIdx]) / (2 * dy)
vorticity = interp_on_time(np.squeeze((dvdx - dudy)[:, 0]), glorysDate, moorDate)
divergence = interp_on_time(np.squeeze((dudx + dvdy)[:, 0]), glorysDate, moorDate)


plt.figure(1, figsize=(10, 6))
plt.plot(dateForPlot, vorticity)
plt.title('vorticity (dv/dx - du/dy)')
plt.savefig(r'figures\check_vorticity_GLORYS.jpg', dpi=300)


plt.figure(2, figsize=(10, 6))
plt.plot(dateForPlot, divergence)
plt.title('divergence (du/dx + dv/dy)')
plt.savefig(r'figures\check_divergence_GLORYS.jpg', dpi=300)
