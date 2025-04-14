import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from astropy.time import Time
from datetime import datetime, timedelta
import scipy.interpolate as itp


def interp_on_time(var, date1, date2):
    itp_t = itp.interp1d(date1, var)
    return itp_t(date2)


# ---------- for my mooring ----------
aviso = nc.Dataset(r'L:\graduation proj\data\AVISO\cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D_1730875116530.nc')
lat = aviso.variables['latitude'][:]
lon = aviso.variables['longitude'][:]
time = aviso.variables['time'][:]
avisoDate = time / 86400 + Time(datetime(1970, 1, 1)).jd - 1721424.5 + 367
u_geo = aviso.variables['ugos'][:]
v_geo = aviso.variables['vgos'][:]
moorData = np.load(r'MoorData/ADCP_uv.npz')
moorDate = moorData['mtime_adcp']
dateForPlot = [datetime(1, 1, 1) + timedelta(days=m-367) for m in moorDate]
lat_moor = 36.23
lon_moor = -32.75
fi = 2 * 7.292e-5 * np.sin(lat_moor/180*np.pi)
# lat: [36.1875 36.3125] 0.0425 0.0825
# lon: [-32.8125 -32.6875] 0.0625 0.0625
# lat: [36.125 36.375] 0.105 0.145
# lon: [-32.875 -32.625] 0.125 0.125
# # ---------- for mooring ICE ----------
# aviso = nc.Dataset(r'L:\graduation proj\data\AVISO\cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_20150601-20170731_-27--29_56_58.nc')
# lat = aviso.variables['latitude'][:]
# lon = aviso.variables['longitude'][:]
# time = aviso.variables['time'][:]
# avisoDate = time / 86400 + Time(datetime(1970, 1, 1)).jd - Time(datetime(1950, 1, 1)).jd
# u_geo = aviso.variables['ugos'][:]
# v_geo = aviso.variables['vgos'][:]
# moorData = np.load(r'mooringICE\mooringICE.npz')
# moorDate = moorData['mtime'][:, 0]
# dateForPlot = [datetime(1950, 1, 1) + timedelta(days=m) for m in moorDate]
# lat_moor = moorData['latitude']
# lon_moor = moorData['longitude']
# fi = 2 * 7.292e-5 * np.sin(lat_moor/180*np.pi)
# lat: [57.5625 57.6875]
# lon: [-28.4375 -28.5625]
# ---------- calculate the vorticity and strain of 4 points ----------
R = 6371000
dxs = R * np.radians(0.125)
# # for mooring ICE
# dys = [R * np.cos(np.radians(i)) * np.radians(0.25) for i in [57.5625, 57.5625, 57.6875, 57.6875]]
# selections = [[-28.4375, 57.5625], [-28.5625, 57.5625], [-28.4375, 57.6875], [-28.5625, 57.6875]]
# for my mooring
dys = [R * np.cos(np.radians(i)) * np.radians(0.25) for i in [36.375, 36.375, 36.125, 36.125]]
selections = [[-32.875, 36.375], [-32.625, 36.375], [-32.875, 36.125], [-32.625, 36.125]]
vorticitySelections = {}
strainSelections = {}
for locs in selections:
    idx = selections.index(locs)
    lonIdx = np.argwhere(lon == locs[0])
    latIdx = np.argwhere(lat == locs[1])
    dudx = (u_geo[:, latIdx, lonIdx + 1] - u_geo[:, latIdx, lonIdx - 1]) / (2 * dxs)
    dudy = (u_geo[:, latIdx + 1, lonIdx] - u_geo[:, latIdx - 1, lonIdx]) / (2 * dys[idx])
    dvdx = (v_geo[:, latIdx, lonIdx + 1] - v_geo[:, latIdx, lonIdx - 1]) / (2 * dxs)
    dvdy = (v_geo[:, latIdx + 1, lonIdx] - v_geo[:, latIdx - 1, lonIdx]) / (2 * dys[idx])
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


