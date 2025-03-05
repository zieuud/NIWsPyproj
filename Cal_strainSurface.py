from datetime import datetime, timedelta
import netCDF4 as nc
import numpy as np
import scipy.interpolate as itp
from astropy.time import Time

aviso = nc.Dataset(r'K:\grad_proj\AVISO\cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D_1730875116530.nc')  # daily
# print(aviso.variables)
lat = aviso.variables['latitude'][:]
lon = aviso.variables['longitude'][:]
time = aviso.variables['time'][:]
# sla = aviso.variables['sla']
u_geo = aviso.variables['ugos'][:]
v_geo = aviso.variables['vgos'][:]  # (time, lat, lon)
lat_moor = 36.23
lon_moor = -32.75
moorData = np.load('ADCP_uv.npz')
moorDate = moorData['mtime']
# ----------calculate the strain----------
dx = 110000./4.
dy = 110000./4.  # 单位m, 分辨率0.25度
latIdx = np.argwhere(lat == 36.125)
lonIdx = np.argwhere(lon == -32.875)
dudx = (u_geo[:, latIdx, lonIdx + 1] - u_geo[:, latIdx, lonIdx - 1]) / (2 * dx)
dudy = (u_geo[:, latIdx + 1, lonIdx] - u_geo[:, latIdx - 1, lonIdx]) / (2 * dy)
dvdx = (v_geo[:, latIdx, lonIdx + 1] - v_geo[:, latIdx, lonIdx - 1]) / (2 * dx)
dvdy = (v_geo[:, latIdx + 1, lonIdx] - v_geo[:, latIdx - 1, lonIdx]) / (2 * dy)
strain_moor = np.squeeze(np.sqrt((dudx - dvdy) ** 2 + (dudy - dvdx) ** 2))
# ---------- interpolate on moor time ----------
avisoDate = time / 86400 + Time(datetime(1970, 1, 1)).jd - 1721424.5 + 366
itp_t = itp.interp1d(avisoDate, strain_moor)
strain_moor_hourly = itp_t(moorDate)
# ---------- save data ----------
np.save(r'ReanaData\AVISO_strain4.npy', strain_moor_hourly)
