import matplotlib.pyplot as plt
import numpy as np
from func_2_cal_stratification import cal_stratification
from func_1_dynmodes import dynmodes
import netCDF4 as nc
import scipy.interpolate as itp
from datetime import datetime, timedelta


lat_moor = 36.23
lon_moor = -32.75
t1 = nc.Dataset(r'L:\graduation proj\data\WOA23\woa23_B5C2_t12_04.nc')
lon_idx = np.argmin(np.abs(t1.variables['lon'][:] - (-32.75)))
lat_idx = np.argmin(np.abs(t1.variables['lat'][:] - 36.23))
lon = t1.variables['lon'][lon_idx]
lat = t1.variables['lat'][lat_idx]
z = -t1.variables['depth'][:]
t = np.zeros((12, 57))
s = np.zeros((12, 57))
for i in range(1, 13):
    path1 = r'L:\graduation proj\data\WOA23\woa23_B5C2_t{:02}_04.nc'.format(i)
    path2 = r'L:\graduation proj\data\WOA23\woa23_B5C2_s{:02}_04.nc'.format(i)
    temp = nc.Dataset(path1)
    t[i-1] = temp.variables['t_an'][0, :, lat_idx, lon_idx]
    salt = nc.Dataset(path2)
    s[i-1] = salt.variables['s_an'][0, :, lat_idx, lon_idx]
ct, sig0, Nsq, dtdz, ze = cal_stratification(12, len(z), z, s, t, lat_moor, lon_moor)

pmodes = np.empty((12, 11, 56))
for t in range(12):
    _, pmodes[t, :, :], _ = dynmodes(Nsq[t, :], ze, 11)

adcp = np.load(r'MoorData/ADCP_uv_ni_byYu.npz')
time = adcp['mtime_adcp']
date = [datetime(1, 1, 1) + timedelta(days=m - 366) for m in time]
depth = adcp['depth_adcp']
adcp2 = np.load(r'MoorData/ADCP_uv_ni_wkb.npz')
u = adcp2['u_ni_wkb']
v = adcp2['v_ni_wkb']
pmodes_moor = np.empty((12, 11, 245))
for t in range(12):
    for m in range(11):
        itp_t = itp.interp1d(ze, pmodes[t, m, :], fill_value=np.nan, bounds_error=False)
        pmodes_moor[t, m, :] = itp_t(depth)

u = u[:, :181]
v = v[:, :181]
depth = depth[:181]
pmodes_moor = pmodes_moor[:, :, :181]
u_mod = np.empty((6650, 11, 181))
v_mod = np.empty((6650, 11, 181))
ke_mod = np.empty((6650, 11, 181))
for t in range(6650):
    m = date[t].month
    valid_indices = np.where(~np.isnan(u[t, :]))[0]
    u_mod_coeff = np.linalg.lstsq(pmodes_moor[m-1, :, valid_indices], u[t, valid_indices], rcond=None)[0]
    u_mod[t, :, valid_indices] = u_mod_coeff * pmodes_moor[m-1, :, valid_indices]
    v_mod_coeff = np.linalg.lstsq(pmodes_moor[m-1, :, valid_indices], v[t, valid_indices], rcond=None)[0]
    v_mod[t, :, valid_indices] = v_mod_coeff * pmodes_moor[m-1, :, valid_indices]
    ke_mod[t, :, :] = 1 / 2 * 1025 * (u_mod[t, :, :] ** 2 + v_mod[t, :, :] ** 2)

ke_mod_dita = np.nanmean(np.nansum(ke_mod, -1), 0)
plt.bar(range(11), ke_mod_dita)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.xlabel('modes')
plt.ylabel('$KE_{ni}^{wkb}$ $(J·m^{-3})$')
plt.show()
print('c')
