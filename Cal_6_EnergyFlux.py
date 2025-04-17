import numpy as np
from matplotlib import pyplot as plt
import scipy.interpolate as itp
from func_0_filter import filter_lp, filter_ni, filter_vlp


moorData = np.load('MoorData/ADCP_uv.npz')
u = np.transpose(moorData['u'].T)[:, 9:181]
v = np.transpose(moorData['v'].T)[:, 9:181]
depthMoor = moorData['depth_adcp'][9:181]
timeMoor = moorData['mtime_adcp']
nt = len(timeMoor)
nz = len(depthMoor)
dt = 3600
dz = 8.
H = -depthMoor[-1]

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
    itp_t = itp.interp1d(ze, Nsq[i, :], fill_value=Nsq[i, -1], bounds_error=False)
    NsqMoor[i, :] = itp_t(depthMoor)

# ---------- calculate the uv perturbation ----------
u_lp = filter_lp(u, dt, nt, lat_moor)
v_lp = filter_lp(v, dt, nt, lat_moor)
ubar0 = np.nansum((u - u_lp) * dz, 1) / H
vbar0 = np.nansum((v - v_lp) * dz, 1) / H
up = u - u_lp - np.tile(ubar0, (nz, 1)).T
vp = v - v_lp - np.tile(vbar0, (nz, 1)).T
up_ni = filter_ni(up, dt, nt, lat_moor)
vp_ni = filter_ni(vp, dt, nt, lat_moor)
# ---------- calculate the pressure perturbation ----------
# calculate the displacement
temp_vlp = filter_vlp(tempMoor, dt, nt, lat_moor)
tempp = tempMoor - temp_vlp
x = -tempp / dtdzMoor
x_ni = filter_ni(x, dt, nt, lat_moor)
# calculate the rho prime
rhop = ((sig0Moor + 1000) / g) * NsqMoor * x
rhop_ni = ((sig0Moor + 1000) / g) * NsqMoor * x_ni

# calculate the p prime
p = np.zeros((nt, nz))
p_ni = np.zeros((nt, nz))
for i in range(nz):
    p[:, i] = np.nansum(rhop[:, :i+1] * g * dz, 1)
    p_ni[:, i] = np.nansum(rhop_ni[:, :i+1] * g * dz, 1)

pbar = np.nansum(p * dz, 1) / H
pbar = -np.tile(pbar, (nz, 1)).T
pp = pbar + p

pbar_ni = np.nansum(p_ni * dz, 1) / H
pbar_ni = -np.tile(pbar_ni, (nz, 1)).T
pp_ni = pbar_ni + p_ni
pp_ni = filter_ni(pp_ni, dt, nt, lat_moor)

fx = pp * up_ni
fy = pp * vp_ni
f = np.sqrt(fx ** 2 + fy ** 2)
f_di = np.nansum(f * dz, 1)

fx_ni = pp_ni * up_ni
fy_ni = pp_ni * vp_ni
f_ni = np.sqrt(fx_ni ** 2 + fy_ni ** 2)
f_ni_di = np.nansum(f_ni * dz, 1)

plt.figure(1)
plt.plot(timeMoor, f_di, 'r-')
plt.plot(timeMoor, f_ni_di, 'k-')

plt.figure(2)
plt.subplot(2, 1, 1)
plt.pcolormesh(timeMoor, depthMoor, f.T, vmin=0, vmax=100)
plt.colorbar()
plt.subplot(2, 1, 2)
plt.pcolormesh(timeMoor, depthMoor, f_ni.T, vmin=0, vmax=100)
plt.colorbar()
# plt.show()

# ---------- save energy flux ----------
np.savez(r'MoorData/EnergyFlux.npz', pp=pp_ni, up=up_ni, vp=vp_ni, fx=fx_ni, fy=fy_ni)

print('c')
