import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from astropy.time import Time
from datetime import datetime, timedelta
import scipy.interpolate as itp


def interp_on_time(var, date1, date2):
    itp_t = itp.interp1d(date1, var)
    return itp_t(date2)


aviso = nc.Dataset(r'K:\grad_proj\AVISO\cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D_1730875116530.nc')
lat = aviso.variables['latitude'][:]
lon = aviso.variables['longitude'][:]
time = aviso.variables['time'][:]
avisoDate = time / 86400 + Time(datetime(1970, 1, 1)).jd - 1721424.5 + 366
u_geo = aviso.variables['ugos'][:]
v_geo = aviso.variables['vgos'][:]
lat_moor = 36.23
lon_moor = -32.75
fi = 2 * 7.292e-5 * np.sin(lat_moor/180*np.pi)
moorData = np.load('ADCP_uv.npz')
moorDate = moorData['mtime']
dateForPlot = [datetime(1, 1, 1) + timedelta(days=m-366) for m in moorDate]
# ---------- calculate the vorticity and strain of 4 points ----------
R = 6371000
dxs = R * np.radians(0.25)
dys = [R * np.cos(np.radians(i)) * np.radians(0.25) for i in [36.375, 36.375, 36.125, 36.125]]
selections = [[-32.625, 36.375], [-32.875, 36.375], [-32.625, 36.125], [-32.875, 36.125]]
vorticitySelections = {}
strainSelections = {}
for locs in selections:
    idx = selections.index(locs)
    lonIdx = np.argwhere(lon == locs[0])
    latIdx = np.argwhere(lat == locs[1])
    dvdx = (v_geo[:, latIdx, lonIdx + 1] - v_geo[:, latIdx, lonIdx - 1]) / (2 * dxs)
    dudy = (u_geo[:, latIdx + 1, lonIdx] - u_geo[:, latIdx - 1, lonIdx]) / (2 * dys[idx])
    dudx = (u_geo[:, latIdx + 1, lonIdx] - u_geo[:, latIdx - 1, lonIdx]) / (2 * dxs)
    dvdy = (v_geo[:, latIdx, lonIdx + 1] - v_geo[:, latIdx, lonIdx - 1]) / (2 * dys[idx])
    vorticity = np.squeeze(dvdx - dudy)
    vorticitySelections[tuple(locs)] = interp_on_time(vorticity, avisoDate, moorDate)
    strain = np.squeeze(np.sqrt((dudx - dvdy) ** 2 + (dudy + dvdx) ** 2))
    strainSelections[tuple(locs)] = interp_on_time(strain, avisoDate, moorDate)
# ---------- save vorticity and strain ----------
num = 1
for vorticity in vorticitySelections.values():
    path = r'ReanaData\AVISO_vorticity{}.npy'.format(num)
    np.save(path, vorticity)
    num += 1
num = 1
for strain in strainSelections.values():
    path = r'ReanaData\AVISO_strain{}.npy'.format(num)
    np.save(path, strain)
    num += 1

print('c')


