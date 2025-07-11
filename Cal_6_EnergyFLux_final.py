import numpy as np
from matplotlib import pyplot as plt
import scipy.interpolate as itp
from func_0_filter import filter_lp, filter_ni, filter_vlp


uv_mod = np.load(r'MoorData/ADCP_uv_ni_10bcmodes.npz')
u_mod = uv_mod['u_mod']
v_mod = uv_mod['v_mod']

endIdx = 181  # 181
# load current data
moorData = np.load('MoorData/ADCP_uv.npz')
depthMoor = moorData['depth_adcp']
timeMoor = moorData['mtime_adcp']
nt = len(timeMoor)
nzMoor = len(depthMoor)
dt = 3600
dz = 8.
H = -depthMoor[-1]

depthFlux = depthMoor[9:endIdx]
nzFlux = len(depthFlux)
# load stratification data
stratification = np.load(r'ReanaData/WOA23_stratification_tempBySensor.npz')
temp = stratification['ct'][:, :36]
sig0 = stratification['sig0'][:, :36]
dtdz = stratification['dtdz'][:, :35]
Nsq = stratification['Nsq'][:, :35]
depthSensor = np.load(r'MoorData/SENSOR_temp.npz')['depth_sensor'][:36]
ze = stratification['ze'][:35]
lat_moor = 36.23
lon_moor = -32.75
g = 9.81
# load modes data
pmodes = np.load(r'ReanaData/WOA23_pmodes_moorGrid_yearly.npz')['pmodes']
pmodesFlux = pmodes[:, 9:endIdx]
nmodes = 11

# ---------- interpolate on moor depth ----------
tempMoor = np.empty((nt, nzFlux))
sig0Moor = np.empty((nt, nzFlux))
dtdzMoor = np.empty((nt, nzFlux))
NsqMoor = np.empty((nt, nzFlux))
for i in range(nt):
    itp_t = itp.interp1d(depthSensor, temp[i, :])
    tempMoor[i, :] = itp_t(depthFlux)
    itp_t = itp.interp1d(depthSensor, sig0[i, :])
    sig0Moor[i, :] = itp_t(depthFlux)
    itp_t = itp.interp1d(ze, dtdz[i, :], fill_value=dtdz[i, -1], bounds_error=False)
    dtdzMoor[i, :] = itp_t(depthFlux)
    itp_t = itp.interp1d(ze, Nsq[i, :], fill_value=Nsq[i, -1], bounds_error=False)
    NsqMoor[i, :] = itp_t(depthFlux)

# ---------- using the uv_ni_mod, remove the bt mode ----------
up_mod_ni = u_mod[:, 1:, 9:endIdx]
vp_mod_ni = v_mod[:, 1:, 9:endIdx]
# ---------- calculate the pressure perturbation ----------
# calculate the epsilon
temp_vlp = filter_vlp(tempMoor, dt, nt, lat_moor)
tempp = tempMoor - temp_vlp
x = -tempp / dtdzMoor
x_ni = filter_ni(x, dt, nt, lat_moor)
# x_ni_mod = np.empty((nt, nmodes, nz))
# for t in range(nt):
#     valid_indices = np.where(~np.isnan(x_ni[t, :]))[0]
#     x_ni_mod_coeff = np.linalg.lstsq(pmodesFlux[t, :, valid_indices], x_ni[t, valid_indices], rcond=None)[0]
#     x_ni_mod[t, :, valid_indices] = x_ni_mod_coeff * pmodesFlux[t, :, valid_indices]
# calculate the rho prime
rhop_ni = ((sig0Moor + 1000) / g) * NsqMoor * x_ni
# calculate the p prime
p_ni = np.zeros((nt, nzFlux))
for i in range(nzFlux):
    p_ni[:, i] = np.nansum(rhop_ni[:, :i + 1] * g * dz, 1)
pbar_ni = np.nansum(p_ni * dz, 1) / H
pbar_ni = -np.tile(pbar_ni, (nzFlux, 1)).T
pp_ni = pbar_ni + p_ni
pp_ni = filter_ni(pp_ni, dt, nt, lat_moor)
pp_ni_mod = np.transpose(np.tile(pp_ni, (nmodes - 1, 1, 1)), (1, 0, 2))
fx_ni_mod = pp_ni_mod * up_mod_ni
fy_ni_mod = pp_ni_mod * vp_mod_ni

np.savez(r'MoorData/EnergyFlux_modes.npz', fx_mod=fy_ni_mod, fy_mod=fx_ni_mod)

