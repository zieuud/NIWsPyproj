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
z = -t1.variables['depth'][:]
s = np.zeros((12, 57))
for i in range(1, 13):
    path2 = r'L:\graduation proj\data\WOA23\woa23_B5C2_s{:02}_04.nc'.format(i)
    salt = nc.Dataset(path2)
    s[i-1] = salt.variables['s_an'][0, :, lat_idx, lon_idx]

sensor = np.load(r'MoorData/SENSOR_temp.npz')
time = sensor['mtime_sensor']
date = [datetime(1, 1, 1) + timedelta(days=m - 366) for m in time]
depth = sensor['depth_sensor']
temp_sensor = sensor['temp']
salt_sensor = np.empty((6650, 40))
for t in range(6650):
    m = date[t].month
    itp_t = itp.interp1d(z, s[m-1, :], fill_value=np.nan, bounds_error=False)
    salt_sensor[t, :] = itp_t(depth)
depth = depth[:36]
temp_sensor = temp_sensor[:, :36]
salt_sensor = salt_sensor[:, :36]
ct, sig0, Nsq, dtdz, ze = cal_stratification(6650, 36, depth, salt_sensor, temp_sensor, lat_moor, lon_moor)

pmodes = np.empty((6650, 11, 35))
for t in range(12):
    _, pmodes[t, :, :], _ = dynmodes(Nsq[t, :], ze, 11)

adcp = np.load(r'MoorData/ADCP_uv_ni_byYu.npz')
depth_adcp = adcp['depth_adcp']
adcp = np.load(r'MoorData/ADCP_uv_ni_wkb.npz')
u = adcp['u_ni_wkb']
v = adcp['v_ni_wkb']

pmodes_moor = np.empty((6650, 11, 245))
for t in range(6650):
    for m in range(11):
        itp_t = itp.interp1d(ze, pmodes[t, m, :], fill_value=np.nan, bounds_error=False)
        pmodes_moor[t, m, :] = itp_t(depth_adcp)

u = u[:, 9:176]
v = v[:, 9:176]
depth = depth[9:176]
pmodes_moor = pmodes_moor[:, :, 9:176]
u_mod = np.empty((6650, 11, 167))
v_mod = np.empty((6650, 11, 167))
ke_mod = np.empty((6650, 11, 167))
for t in range(6650):
    valid_indices = np.where(~np.isnan(u[t, :]))[0]
    u_mod_coeff = np.linalg.lstsq(pmodes_moor[t, :, valid_indices], u[t, valid_indices], rcond=None)[0]
    u_mod[t, :, valid_indices] = u_mod_coeff * pmodes_moor[t, :, valid_indices]
    v_mod_coeff = np.linalg.lstsq(pmodes_moor[t, :, valid_indices], v[t, valid_indices], rcond=None)[0]
    v_mod[t, :, valid_indices] = v_mod_coeff * pmodes_moor[t, :, valid_indices]
    ke_mod[t, :, :] = 1 / 2 * 1025 * (u_mod[t, :, :] ** 2 + v_mod[t, :, :] ** 2)

ke_mod_dita = np.nanmean(np.nansum(ke_mod, -1), 0)
plt.bar(range(11), ke_mod_dita)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.xlabel('modes')
plt.ylabel('$KE_{ni}^{wkb}$ $(J·m^{-3})$')
plt.show()
print('c')
