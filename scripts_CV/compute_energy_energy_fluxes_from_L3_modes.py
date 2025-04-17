"""
CV 2019/11/25: compute energy fluxes from mooring data, u and xi are first
               projected onto the modes, following Nash et al 2005:
               << the determination of baroclinic pressure anomaly requires full-depth,
               continous data. Because neither type of moored dataset satisfies this requirement,
               we rely on normal modes to generate the full-depth profiles. >>
CV 2019/11/27: also computes energy fluxes with non-modal variables.
               ------> THIS MAKES compute_energy_energy_fluxes_modes.py obsolete <------
"""
from rrex_moorings import RREX_moorings
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
import matplotlib.colors as colors
import matplotlib.dates as dates
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import scipy.signal as sig
import scipy.interpolate as itp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import gsw as gsw
from netCDF4 import Dataset
import sys

sys.path.append('../../../Python_miscellaneous/')
from dynmodes_bt import dynmodes

# ------------ parameters ------------ 
path_data = '/Users/cv1m15/Data/'
mooring = 'IRW'
H = 2063 + 50  #  mooring depth

# ti,tf = 23920,23980
# ti,tf = 23920,23920+60
ti, tf = 23920, 24674  # works for all moorings except ICW
# ti,tf = 23920,24629  # ICW
g = 9.81
nmodes = 6
file_out = path_data + mooring + '_fluxes_from_%.2i_modes.nc' % nmodes

# ------------ read data ------------
data = RREX_moorings(list_moorings=[mooring], time=[ti, tf], level='L3')
# data = RREX_moorings(time=[ti,tf])
data.get_date()

# ------------ processing ------------ 
print(' ... processing ... ')
print('######################## 0. COMPUTE THE MODES ##########################################')
# --- define regular grid onto which all variables are projected --- 
dz = 20
zz = np.arange(-H, dz, dz)
nz = zz.shape[0]

# --- use N2 from monthly WOA --- 
data.get_N2_from_WOA()
N2_grid = np.zeros((data.nt, nz))
sig0_grid = np.zeros((data.nt, nz))
for t in range(data.nt):
    fitp = itp.interp1d(data.ze_woa, data.N2_woa_hourly[t, :], fill_value='extrapolate')  # extrapolate at the surface
    N2_grid[t, :] = fitp(zz)
    fitp = itp.interp1d(data.zc_woa, data.sig0_woa_hourly[t, :], fill_value='extrapolate')
    sig0_grid[t, :] = fitp(zz)

# plt.figure()
# plt.plot(data.z_uv[:,0],'k',lw=0.4)
# plt.plot(data.z_uv[:,1],'k',lw=0.4)
# plt.plot(data.z_uv[:,2],'k',lw=0.4)
# plt.xlim(16500,18096)
# plt.savefig('tmp.pdf')
# exit()

# --- compute the modes ---
print(' ... compute the modes at every time step ... ')
pmodes = np.zeros((data.nt, nmodes, nz))
wmodes = np.zeros((data.nt, nmodes, nz))
pmodes_zuv = np.zeros((data.nt, nmodes, data.nz_uv))  # pmodes at u,v points
wmodes_zts = np.zeros((data.nt, nmodes, data.nz_ts))  # wmodes at t,s points
for t in range(data.nt):
    print('%.5i / %.5i --- %.f --- %.f --- %.f' % (t, data.nt, data.z_uv[t, -2], data.z_uv[t, -1], data.z_uv[t, 0]))
    [wmodes[t, :, :], pmodes[t, :, :], _] = dynmodes(N2_grid[t, :], zz, nmodes)
    for m in range(nmodes):
        fitp = itp.interp1d(zz, pmodes[t, m, :])
        pmodes_zuv[t, m, :] = fitp(data.z_uv[t, :].data)
        fitp = itp.interp1d(zz, wmodes[t, m, :])
        wmodes_zts[t, m, :] = fitp(data.z_ts[t, :].data)

print('######################## 1. COMPUTE VELOCITY PERTURBATION ##############################')
# --- remove low-pass velocity --- 
u_lp = data.time_filtering(data.u, filter_type='lp')
v_lp = data.time_filtering(data.v, filter_type='lp')
u = data.u - u_lp  # u prime
v = data.v - v_lp

###################### METHOD: WITH MODAL FIT ############################
# --- project onto the modes --- 
u_cmod = np.zeros((data.nt, nmodes))  # modal coefficients
v_cmod = np.zeros((data.nt, nmodes))  # modal coefficients
u_mod = np.zeros((data.nt, nmodes, nz))  # projected u onto grid
v_mod = np.zeros((data.nt, nmodes, nz))  # projected u onto grid
for t in range(data.nt):
    u_tmp = np.copy(u[t, :]);
    kgood = np.argwhere(~np.isnan(u_tmp))
    u_cmod[t, :] = np.linalg.lstsq(np.squeeze(pmodes_zuv[t, :, kgood]), np.squeeze(u[t, kgood]))[0]
    u_mod[t, :, :] = (u_cmod[t, :] * pmodes[t, :, :].T).T
    v_tmp = np.copy(v[t, :]);
    kgood = np.argwhere(~np.isnan(v_tmp))
    v_cmod[t, :] = np.linalg.lstsq(np.squeeze(pmodes_zuv[t, :, kgood]), np.squeeze(v[t, kgood]))[0]
    v_mod[t, :, :] = (v_cmod[t, :] * pmodes[t, :, :].T).T

# --- baroclinicity condition --- 
# - just remove the barotropic contribution - 
u_mod_bc = np.nansum(u_mod[:, 1:, :], axis=1)  # perturbation signal
v_mod_bc = np.nansum(v_mod[:, 1:, :], axis=1)  # perturbation signal

# --- filtering --- 
u_mod_sd = data.time_filtering(u_mod_bc, filter_type='sd')
v_mod_sd = data.time_filtering(v_mod_bc, filter_type='sd')
u_mod_ni = data.time_filtering(u_mod_bc, filter_type='ni')
v_mod_ni = data.time_filtering(v_mod_bc, filter_type='ni')
u_mod_iw = data.time_filtering(u_mod_bc, filter_type='iw')
v_mod_iw = data.time_filtering(v_mod_bc, filter_type='iw')

###################### METHOD: NO   MODAL FIT ############################
# --- baroclinicity condition --- 
ze_uv = 0.5 * (data.z_uv[:, 1:] + data.z_uv[:, :-1])
dz_uv = np.diff(ze_uv, axis=1)
dz_uv = np.concatenate(([dz_uv[:, 0]], dz_uv.T, [dz_uv[:, -1]]), axis=0).T
u = u - np.tile(np.nansum(u * dz_uv, axis=1) / np.nansum(dz_uv, axis=1), (data.nz_uv, 1)).T
v = v - np.tile(np.nansum(v * dz_uv, axis=1) / np.nansum(dz_uv, axis=1), (data.nz_uv, 1)).T

# --- filtering --- 
u_sd = data.time_filtering(u, filter_type='sd')
v_sd = data.time_filtering(v, filter_type='sd')
u_ni = data.time_filtering(u, filter_type='ni')
v_ni = data.time_filtering(v, filter_type='ni')
u_iw = data.time_filtering(u, filter_type='iw')
v_iw = data.time_filtering(v, filter_type='iw')

print('######################## 2. COMPUTE PRESSURE PERTURBATION ##############################')
# --- remove low-pass temperature --- 
t_vlp = data.time_filtering(data.t, filter_type='vlp')  # very low-pass
t_lp = data.time_filtering(data.t, filter_type='lp')  # low-pass
tp = data.t - t_lp

# plt.figure()
# for k in range(data.nz_ts):
#    plt.plot(data.dayad,data.t[:,k],'k',lw=0.4) 
#    plt.plot(data.dayad,t_lp[:,k],'r',lw=0.8) 
#    plt.plot(data.dayad,t_vlp[:,k],'g',lw=0.8) 
# plt.xlim(0,100)
# plt.savefig('tmp.pdf')

# --- compute temperature gradient from very low-pass data --- 
data.ze_ts = 0.5 * (data.z_ts[:, 1:] + data.z_ts[:, :-1])
dtdz = np.diff(t_vlp, axis=1) / np.diff(data.z_ts, axis=1)
dtdz_zi = np.zeros((data.nt, data.nz_ts))
for t in range(data.nt):
    itp_dtdz = itp.interp1d(data.ze_ts[t, :], dtdz[t, :], kind='linear', fill_value='extrapolate')
    dtdz_zi[t, :] = itp_dtdz(data.z_ts[t, :].data)
dtdz_zi[dtdz_zi < 1e-4] = 1e-4  # lower bound in very weakly stratified waters where dT/dz goes to zero

# - quick check on dtdz from thermistors vs climatology - 
# plt.figure()
# plt.plot(data.dayad,dtdz_zi[:,1],'k',lw=0.3)
# plt.plot(data.dayad,data.dtdz_woa_hourly_zts[:,1],'k--',lw=0.3)
# plt.plot(data.dayad,dtdz_zi[:,2],'r',lw=0.3)
# plt.plot(data.dayad,data.dtdz_woa_hourly_zts[:,2],'r--',lw=0.3)
# plt.plot(data.dayad,dtdz_zi[:,3],'g',lw=0.3)
# plt.plot(data.dayad,data.dtdz_woa_hourly_zts[:,3],'g--',lw=0.3)
# plt.plot(data.dayad,dtdz_zi[:,4],'y',lw=0.3)
# plt.plot(data.dayad,data.dtdz_woa_hourly_zts[:,4],'y--',lw=0.3)
# plt.savefig('tmp.pdf')

# --- compute displacement ---
# x = -tp/dtdz_zi                 # --> from gradient computed using thermistors' data
x = -tp / data.dtdz_woa_hourly_zts  # --> from gradient computed with climatological data

# --- filtering --- 
x_sd = data.time_filtering(x, filter_type='sd')
x_ni = data.time_filtering(x, filter_type='ni')
x_iw = data.time_filtering(x, filter_type='iw')

# --- project onto the modes --- 
x_cmod = np.zeros((data.nt, nmodes))  # modal coefficients
x_sd_cmod = np.zeros((data.nt, nmodes))
x_ni_cmod = np.zeros((data.nt, nmodes))
x_iw_cmod = np.zeros((data.nt, nmodes))
x_mod = np.zeros((data.nt, nmodes, nz))  # projected xi onto grid
x_sd_mod = np.zeros((data.nt, nmodes, nz))
x_ni_mod = np.zeros((data.nt, nmodes, nz))
x_iw_mod = np.zeros((data.nt, nmodes, nz))
for t in range(data.nt):
    x_tmp = np.copy(x[t, :]);
    kgood = np.argwhere(~np.isnan(x_tmp))
    x_cmod[t, :] = np.linalg.lstsq(np.squeeze(wmodes_zts[t, :, kgood]), np.squeeze(x[t, kgood]))[0]
    x_mod[t, :, :] = (x_cmod[t, :] * wmodes[t, :, :].T).T
    x_sd_cmod[t, :] = np.linalg.lstsq(np.squeeze(wmodes_zts[t, :, kgood]), np.squeeze(x_sd[t, kgood]))[0]
    x_sd_mod[t, :, :] = (x_sd_cmod[t, :] * wmodes[t, :, :].T).T
    x_ni_cmod[t, :] = np.linalg.lstsq(np.squeeze(wmodes_zts[t, :, kgood]), np.squeeze(x_ni[t, kgood]))[0]
    x_ni_mod[t, :, :] = (x_ni_cmod[t, :] * wmodes[t, :, :].T).T
    x_iw_cmod[t, :] = np.linalg.lstsq(np.squeeze(wmodes_zts[t, :, kgood]), np.squeeze(x_iw[t, kgood]))[0]
    x_iw_mod[t, :, :] = (x_iw_cmod[t, :] * wmodes[t, :, :].T).T

# --- compute pressure anomaly --- 
###################### METHOD: WITH MODAL FIT ############################
rhoa = ((sig0_grid + 1000) / g) * N2_grid * np.nansum(x_mod, axis=1)
rhoa_sd = ((sig0_grid + 1000) / g) * N2_grid * np.nansum(x_sd_mod, axis=1)
rhoa_ni = ((sig0_grid + 1000) / g) * N2_grid * np.nansum(x_ni_mod, axis=1)
rhoa_iw = ((sig0_grid + 1000) / g) * N2_grid * np.nansum(x_iw_mod, axis=1)

p_mod = np.zeros((data.nt, nz))
p_sd_mod = np.zeros((data.nt, nz))
p_ni_mod = np.zeros((data.nt, nz))
p_iw_mod = np.zeros((data.nt, nz))
for k in range(nz):
    p_mod[:, k] = np.nansum(rhoa[:, k:] * dz * g, axis=1)
    p_sd_mod[:, k] = np.nansum(rhoa_sd[:, k:] * dz * g, axis=1)
    p_ni_mod[:, k] = np.nansum(rhoa_ni[:, k:] * dz * g, axis=1)
    p_iw_mod[:, k] = np.nansum(rhoa_iw[:, k:] * dz * g, axis=1)

# - baroclinicity condition - 
p_mod = p_mod - np.tile(np.nansum(p_mod * dz, axis=1) / H, (nz, 1)).T
p_sd_mod = p_sd_mod - np.tile(np.nansum(p_sd_mod * dz, axis=1) / H, (nz, 1)).T
p_ni_mod = p_ni_mod - np.tile(np.nansum(p_ni_mod * dz, axis=1) / H, (nz, 1)).T
p_iw_mod = p_iw_mod - np.tile(np.nansum(p_iw_mod * dz, axis=1) / H, (nz, 1)).T

###################### METHOD: NO   MODAL FIT ############################
rhoa = ((data.sig0_woa_hourly_zts + 1000) / g) * data.N2_woa_hourly_zts * x
rhoa_sd = ((data.sig0_woa_hourly_zts + 1000) / g) * data.N2_woa_hourly_zts * x_sd
rhoa_ni = ((data.sig0_woa_hourly_zts + 1000) / g) * data.N2_woa_hourly_zts * x_ni
rhoa_iw = ((data.sig0_woa_hourly_zts + 1000) / g) * data.N2_woa_hourly_zts * x_iw

p = np.zeros((data.nt, data.nz_ts))
p_sd = np.zeros((data.nt, data.nz_ts))
p_ni = np.zeros((data.nt, data.nz_ts))
p_iw = np.zeros((data.nt, data.nz_ts))
dz_ts = np.diff(data.ze_ts, axis=1)
dz_ts = np.concatenate(([dz_ts[:, 0]], dz_ts.T, [dz_ts[:, -1]]), axis=0).T
for k in range(data.nz_ts):
    p[:, k] = np.nansum(rhoa[:, k:] * dz_ts[:, k:] * g, axis=1)
    p_sd[:, k] = np.nansum(rhoa_sd[:, k:] * dz_ts[:, k:] * g, axis=1)
    p_ni[:, k] = np.nansum(rhoa_ni[:, k:] * dz_ts[:, k:] * g, axis=1)
    p_iw[:, k] = np.nansum(rhoa_iw[:, k:] * dz_ts[:, k:] * g, axis=1)

# - baroclinicity condition - 
p = p - np.tile(np.nansum(p * dz_ts, axis=1) / np.nansum(dz_ts, axis=1), (data.nz_ts, 1)).T
p_sd = p_sd - np.tile(np.nansum(p_sd * dz_ts, axis=1) / np.nansum(dz_ts, axis=1), (data.nz_ts, 1)).T
p_ni = p_ni - np.tile(np.nansum(p_ni * dz_ts, axis=1) / np.nansum(dz_ts, axis=1), (data.nz_ts, 1)).T
p_iw = p_iw - np.tile(np.nansum(p_iw * dz_ts, axis=1) / np.nansum(dz_ts, axis=1), (data.nz_ts, 1)).T

print('######################## 3. COMPUTE ENERGY FLUXES ######################################')
###################### METHOD: NO   MODAL FIT ############################
# --- project variables onto common grid --- 
u_itp = np.zeros((data.nt, nz))
v_itp = np.zeros((data.nt, nz))
x_itp = np.zeros((data.nt, nz))  # for plotting purposes
p_itp = np.zeros((data.nt, nz))
u_sd_itp = np.zeros((data.nt, nz))
v_sd_itp = np.zeros((data.nt, nz))
x_sd_itp = np.zeros((data.nt, nz))
p_sd_itp = np.zeros((data.nt, nz))
u_ni_itp = np.zeros((data.nt, nz))
v_ni_itp = np.zeros((data.nt, nz))
x_ni_itp = np.zeros((data.nt, nz))
p_ni_itp = np.zeros((data.nt, nz))
u_iw_itp = np.zeros((data.nt, nz))
v_iw_itp = np.zeros((data.nt, nz))
x_iw_itp = np.zeros((data.nt, nz))
p_iw_itp = np.zeros((data.nt, nz))
for t in range(data.nt):
    fitp = itp.interp1d(data.z_uv[t, :], u[t, :], bounds_error=False)
    u_itp[t, :] = fitp(zz)
    fitp = itp.interp1d(data.z_uv[t, :], v[t, :], bounds_error=False)
    v_itp[t, :] = fitp(zz)
    fitp = itp.interp1d(data.z_ts[t, :], p[t, :], bounds_error=False)
    p_itp[t, :] = fitp(zz)
    # - sd - 
    fitp = itp.interp1d(data.z_uv[t, :], u_sd[t, :], bounds_error=False)
    u_sd_itp[t, :] = fitp(zz)
    fitp = itp.interp1d(data.z_uv[t, :], v_sd[t, :], bounds_error=False)
    v_sd_itp[t, :] = fitp(zz)
    fitp = itp.interp1d(data.z_ts[t, :], x_sd[t, :], bounds_error=False)
    x_sd_itp[t, :] = fitp(zz)
    fitp = itp.interp1d(data.z_ts[t, :], p_sd[t, :], bounds_error=False)
    p_sd_itp[t, :] = fitp(zz)
    # - ni - 
    fitp = itp.interp1d(data.z_uv[t, :], u_ni[t, :], bounds_error=False)
    u_ni_itp[t, :] = fitp(zz)
    fitp = itp.interp1d(data.z_uv[t, :], v_ni[t, :], bounds_error=False)
    v_ni_itp[t, :] = fitp(zz)
    fitp = itp.interp1d(data.z_ts[t, :], x_ni[t, :], bounds_error=False)
    x_ni_itp[t, :] = fitp(zz)
    fitp = itp.interp1d(data.z_ts[t, :], p_ni[t, :], bounds_error=False)
    p_ni_itp[t, :] = fitp(zz)
    # - iw - 
    fitp = itp.interp1d(data.z_uv[t, :], u_iw[t, :], bounds_error=False)
    u_iw_itp[t, :] = fitp(zz)
    fitp = itp.interp1d(data.z_uv[t, :], v_iw[t, :], bounds_error=False)
    v_iw_itp[t, :] = fitp(zz)
    fitp = itp.interp1d(data.z_ts[t, :], x_iw[t, :], bounds_error=False)
    x_iw_itp[t, :] = fitp(zz)
    fitp = itp.interp1d(data.z_ts[t, :], p_iw[t, :], bounds_error=False)
    p_iw_itp[t, :] = fitp(zz)

# then open the netcdf file (next section) and just multiply u,v by p. 

print('######################## 4. SAVE IN NETCDF FILE ########################################')
nc = Dataset(file_out, 'w')
nc.createDimension('time', data.nt)
nc.createDimension('z', nz)
nc.createDimension('z_uv', data.nz_uv)
nc.createDimension('z_ts', data.nz_ts)
nc.createDimension('mode', nmodes)
# - common variable to all grids - 
nc.createVariable('time', 'f', ('time',))
nc.variables['time'][:] = data.time
# - on native uv grid - 
nc.createVariable('z_uv', 'f', ('time', 'z_uv'))
nc.createVariable('u', 'f', ('time', 'z_uv'))
nc.createVariable('v', 'f', ('time', 'z_uv'))
nc.createVariable('u_sd', 'f', ('time', 'z_uv'))
nc.createVariable('v_sd', 'f', ('time', 'z_uv'))
nc.createVariable('u_ni', 'f', ('time', 'z_uv'))
nc.createVariable('v_ni', 'f', ('time', 'z_uv'))
nc.createVariable('u_iw', 'f', ('time', 'z_uv'))
nc.createVariable('v_iw', 'f', ('time', 'z_uv'))
nc.variables['z_uv'][:] = data.z_uv
nc.variables['u'][:] = u
nc.variables['v'][:] = v
nc.variables['u_sd'][:] = u_sd
nc.variables['v_sd'][:] = v_sd
nc.variables['u_ni'][:] = u_ni
nc.variables['v_ni'][:] = v_ni
nc.variables['u_iw'][:] = u_iw
nc.variables['v_iw'][:] = v_iw
# - on native ts grid - 
nc.createVariable('z_ts', 'f', ('time', 'z_ts'))
nc.createVariable('x', 'f', ('time', 'z_ts'))
nc.createVariable('x_sd', 'f', ('time', 'z_ts'))
nc.createVariable('x_ni', 'f', ('time', 'z_ts'))
nc.createVariable('x_iw', 'f', ('time', 'z_ts'))
nc.createVariable('p', 'f', ('time', 'z_ts'))
nc.createVariable('p_sd', 'f', ('time', 'z_ts'))
nc.createVariable('p_ni', 'f', ('time', 'z_ts'))
nc.createVariable('p_iw', 'f', ('time', 'z_ts'))
nc.variables['z_ts'][:] = data.z_ts
nc.variables['x'][:] = x
nc.variables['x_sd'][:] = x_sd
nc.variables['x_ni'][:] = x_ni
nc.variables['x_iw'][:] = x_iw
nc.variables['p'][:] = p
nc.variables['p_sd'][:] = p_sd
nc.variables['p_ni'][:] = p_ni
nc.variables['p_iw'][:] = p_iw
# - on common grid - 
#   --> modes 
nc.createVariable('z', 'f', ('z',))
nc.createVariable('pmodes', 'f', ('time', 'mode', 'z'))
nc.createVariable('wmodes', 'f', ('time', 'mode', 'z'))
nc.variables['z'][:] = zz
nc.variables['pmodes'][:] = pmodes
nc.variables['wmodes'][:] = wmodes
#   --> u,v 
nc.createVariable('u_mod', 'f', ('time', 'mode', 'z'))
nc.createVariable('v_mod', 'f', ('time', 'mode', 'z'))
nc.createVariable('u_mod_sd', 'f', ('time', 'z'))
nc.createVariable('v_mod_sd', 'f', ('time', 'z'))
nc.createVariable('u_mod_ni', 'f', ('time', 'z'))
nc.createVariable('v_mod_ni', 'f', ('time', 'z'))
nc.createVariable('u_mod_iw', 'f', ('time', 'z'))
nc.createVariable('v_mod_iw', 'f', ('time', 'z'))
nc.createVariable('u_itp', 'f', ('time', 'z'))
nc.createVariable('v_itp', 'f', ('time', 'z'))
nc.createVariable('u_sd_itp', 'f', ('time', 'z'))
nc.createVariable('v_sd_itp', 'f', ('time', 'z'))
nc.createVariable('u_ni_itp', 'f', ('time', 'z'))
nc.createVariable('v_ni_itp', 'f', ('time', 'z'))
nc.createVariable('u_iw_itp', 'f', ('time', 'z'))
nc.createVariable('v_iw_itp', 'f', ('time', 'z'))
nc.variables['u_mod'][:] = u_mod
nc.variables['v_mod'][:] = v_mod
nc.variables['u_mod_sd'][:] = u_mod_sd
nc.variables['v_mod_sd'][:] = v_mod_sd
nc.variables['u_mod_ni'][:] = u_mod_ni
nc.variables['v_mod_ni'][:] = v_mod_ni
nc.variables['u_mod_iw'][:] = u_mod_iw
nc.variables['v_mod_iw'][:] = v_mod_iw
nc.variables['u_itp'][:] = u_itp
nc.variables['v_itp'][:] = v_itp
nc.variables['u_sd_itp'][:] = u_sd_itp
nc.variables['v_sd_itp'][:] = v_sd_itp
nc.variables['u_ni_itp'][:] = u_ni_itp
nc.variables['v_ni_itp'][:] = v_ni_itp
nc.variables['u_iw_itp'][:] = u_iw_itp
nc.variables['v_iw_itp'][:] = v_iw_itp
#   --> x,p 
nc.createVariable('x_mod', 'f', ('time', 'mode', 'z'))
nc.createVariable('x_sd_mod', 'f', ('time', 'mode', 'z'))
nc.createVariable('x_ni_mod', 'f', ('time', 'mode', 'z'))
nc.createVariable('x_iw_mod', 'f', ('time', 'mode', 'z'))
nc.createVariable('p_mod', 'f', ('time', 'z'))
nc.createVariable('p_sd_mod', 'f', ('time', 'z'))
nc.createVariable('p_ni_mod', 'f', ('time', 'z'))
nc.createVariable('p_iw_mod', 'f', ('time', 'z'))
nc.createVariable('x_itp', 'f', ('time', 'z'))
nc.createVariable('x_sd_itp', 'f', ('time', 'z'))
nc.createVariable('x_ni_itp', 'f', ('time', 'z'))
nc.createVariable('x_iw_itp', 'f', ('time', 'z'))
nc.createVariable('p_itp', 'f', ('time', 'z'))
nc.createVariable('p_sd_itp', 'f', ('time', 'z'))
nc.createVariable('p_ni_itp', 'f', ('time', 'z'))
nc.createVariable('p_iw_itp', 'f', ('time', 'z'))
nc.variables['x_mod'][:] = x_mod
nc.variables['x_sd_mod'][:] = x_sd_mod
nc.variables['x_ni_mod'][:] = x_ni_mod
nc.variables['x_iw_mod'][:] = x_iw_mod
nc.variables['p_mod'][:] = p_mod
nc.variables['p_sd_mod'][:] = p_sd_mod
nc.variables['p_ni_mod'][:] = p_ni_mod
nc.variables['p_iw_mod'][:] = p_iw_mod
nc.variables['x_itp'][:] = p_itp
nc.variables['x_sd_itp'][:] = p_sd_itp
nc.variables['x_ni_itp'][:] = p_ni_itp
nc.variables['x_iw_itp'][:] = p_iw_itp
nc.variables['p_itp'][:] = p_itp
nc.variables['p_sd_itp'][:] = p_sd_itp
nc.variables['p_ni_itp'][:] = p_ni_itp
nc.variables['p_iw_itp'][:] = p_iw_itp
nc.close()
