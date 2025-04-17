import numpy as np
from matplotlib import pyplot as plt
import scipy.interpolate as itp
from func_0_filter import filter_lp, filter_ni, filter_vlp


moorData = np.load('MoorData/ADCP_uv.npz')
u = np.transpose(moorData['u'].T)
v = np.transpose(moorData['v'].T)
depthMoor = moorData['depth_adcp']
timeMoor = moorData['mtime_adcp']
u_cut = u[:3960, :]
v_cut = v[:3960, :]
timeMoor_cut = timeMoor[:3960]
nt = len(timeMoor_cut)
nzMoor = len(depthMoor)
dt = 3600
dz = 8.
H = -depthMoor[-1]
lat_moor = 36.23
lon_moor = -32.75
pmodes = np.load(r'ReanaData/WOA23_pmodes_moorGrid_yearly.npz')['pmodes'][:6, :]
nmodes = 6

u_lp = filter_lp(u_cut, dt, nt, lat_moor)
v_lp = filter_lp(v_cut, dt, nt, lat_moor)
up = u_cut - u_lp
vp = v_cut - v_lp

up_ni = filter_ni(up, dt, nt, lat_moor)
vp_ni = filter_ni(vp, dt, nt, lat_moor)

up_mod = np.empty((nt, nmodes, nzMoor)) * np.nan
vp_mod = np.empty((nt, nmodes, nzMoor)) * np.nan
for t in range(nt):
    valid_indices = np.where(~np.isnan(up_ni[t, :]))[0]
    up_mod_coeff = np.linalg.lstsq(pmodes[:, valid_indices].T, up_ni[t, valid_indices], rcond=None)[0]
    up_mod[t, :, valid_indices] = (up_mod_coeff.reshape(nmodes, 1) * pmodes[:, valid_indices]).T
    vp_mod_coeff = np.linalg.lstsq(pmodes[:, valid_indices].T, vp_ni[t, valid_indices], rcond=None)[0]
    vp_mod[t, :, valid_indices] = (vp_mod_coeff.reshape(nmodes, 1) * pmodes[:, valid_indices]).T

# plt.figure(1)
# ke_mod = 1/2 * 1025 * (up_mod ** 2 + vp_mod ** 2)
# for i in range(1, 6):
#     plt.subplot(5, 1, i)
#     plt.pcolormesh(timeMoor_cut, depthMoor, ke_mod[:, i, :].T, vmin=0, vmax=15)
#     plt.colorbar()

plt.figure(2)
ke_mod_ni = 1/2 * 1025 * (up_mod ** 2 + vp_mod ** 2)
for i in range(1, 6):
    plt.subplot(5, 1, i)
    plt.pcolormesh(timeMoor_cut, depthMoor, ke_mod_ni[:, i, :].T)
    plt.colorbar()
plt.show()

ke_mod_contribution = np.nanmean(np.nansum(ke_mod_ni, -1), 0)

plt.bar(range(0, 6), ke_mod_contribution)
plt.show()
print('c')
