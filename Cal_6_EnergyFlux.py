import numpy as np
import gsw
import scipy.signal as sig
from matplotlib import pyplot as plt
import scipy.interpolate as itp
from func_0_filter import filter_lp, filter_ni, filter_vlp


moorData = np.load('ADCP_uv.npz')
u = np.transpose(moorData['u'])[:, 9:181]
v = np.transpose(moorData['v'])[:, 9:181]
depthMoor = moorData['depth'][9:181]
timeMoor = moorData['mtime']
nt = len(timeMoor)
nz = len(depthMoor)
dt = 3600
dz = -8.

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

# ---------- interpolate on moor depth ----------
tempMoor = np.empty((nt, nz))
sig0Moor = np.empty((nt, nz))
dtdzMoor = np.empty((nt, nz))
NsqMoor = np.empty((nt, nz))
for i in range(nt):
    itp_t = itp.interp1d(depthSensor, temp[i, :])
    tempMoor[i, :] = itp_t(depthMoor)
    itp_t = itp.interp1d(depthSensor, sig0[i, :])
    sig0Moor[i, :] = itp_t(depthMoor)
    itp_t = itp.interp1d(ze, dtdz[i, :], fill_value=dtdz[i, -1], bounds_error=False)
    dtdzMoor[i, :] = itp_t(depthMoor)
    itp_t = itp.interp1d(ze, Nsq[i, :], fill_value=dtdz[i, -1], bounds_error=False)
    NsqMoor[i, :] = itp_t(depthMoor)

# ---------- calculate the uv perturbation ----------
u_lp = filter_lp(u, dt, nt, lat_moor)
v_lp = filter_lp(v, dt, nt, lat_moor)
ubar0 = np.trapz((u - u_lp), -depthMoor) / -depthMoor[-1]
vbar0 = np.trapz((v - v_lp), -depthMoor) / -depthMoor[-1]
up = u - u_lp - np.tile(ubar0, (nz, 1)).T
vp = v - v_lp - np.tile(vbar0, (nz, 1)).T
up_ni = filter_ni(up, dt, nt, lat_moor)
vp_ni = filter_ni(vp, dt, nt, lat_moor)
# ---------- calculate the pressure perturbation ----------
# calculate the epsilon
temp_vlp = filter_vlp(tempMoor, dt, nt, lat_moor)
tempp = tempMoor - temp_vlp
x = -tempp / dtdzMoor
# x_ni = filter_ni(x, dt, nt, lat_moor)
# calculate the rho prime
rhop = ((np.nanmean(sig0Moor, 0) + 1000) / g) * NsqMoor * x
# rhop_ni = ((sig0 + 1000) / g) * N2 * x_ni
# calculate the p prime
p = np.zeros((nt, nz))
# p_ni = np.zeros((nt, nz))
for i in range(nz):
    p[:, i] = np.nansum(rhop[:, :i + 1], 1) * g * dz
    # p[:, i] = np.trapz(rhop[:, :i+1], -depth[:]) * g
    # p_ni[:, i] = np.trapz(rhop_ni[:, i:], -depth[i:]) * g

pbar = np.nansum(p, 1) / nz
pbar = -np.tile(pbar, (nz, 1)).T
pp = pbar + p
check = np.nansum(pp, 1) * dz
# pbar_ni = -np.trapz(p_ni, -depth) / -depth[-1]
# pp_ni = np.tile(pbar_ni, (nz, 1)).T + p_ni

fx1 = pp * up_ni
# fx = pp_ni * up_ni
# fx_di = np.trapz(fx, depth)
# fy = pp_ni * vp_ni
# ---------- save energy flux ----------
# np.savez(r'ReanaData\WOA23_horizontalEnergyFlux.npz', fx=fx, fy=fy)

print('c')
