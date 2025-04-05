import matplotlib.pyplot as plt
import numpy as np
import func_0_filter as func

# read adcp data
adcp = np.load(r'MoorData\ADCP_uv.npz')
depth = adcp['depth_adcp']
mtime = adcp['mtime_adcp']
lat_moor = 36.23
lon_moor = -32.75
u = adcp['u']
v = adcp['v']
# read snesor data
temp = np.load(r'MoorData\SENSOR_temp_interpolate.npy')
dt = 3600
nt = len(mtime)
dz = depth[1] - depth[0]
nz = len(depth)
g = 9.81
# read woa23 data
Nsq = np.load(r'ReanaData\WOA23_N2_grid.npy')
sig0 = np.load(r'ReanaData\WOA23_sig0_grid.npy')
dtdz = np.load(r'ReanaData\WOA23_dtdz_grid.npy ')

# ---------- calculate the uv perturbation ----------
u_lp = func.filter_lp(u, dt, nt, lat_moor)
v_lp = func.filter_lp(v, dt, nt, lat_moor)
ubar0 = np.nansum((u - u_lp) * dz, 1) / nz / dz
vbar0 = np.nansum((v - v_lp) * dz, 1) / nz / dz
up = u - u_lp - np.tile(ubar0, (nz, 1)).T
vp = v - v_lp - np.tile(vbar0, (nz, 1)).T
up_ni = func.filter_ni(up, dt, nt, lat_moor)
vp_ni = func.filter_ni(vp, dt, nt, lat_moor)
# ---------- calculate the pressure perturbation ----------
# calculate the epsilon
temp_lp = func.filter_lp(temp, dt, nt, lat_moor)
tempp = temp - temp_lp
# dtdz = np.diff(temp) / dz
# dtdz = np.concatenate((dtdz, dtdz[:, -1].reshape(-1, 1)), axis=1)
x = -tempp / dtdz
# calculate the density perturbation
rhop = (sig0 + 1000) / g * Nsq * x
rhop_int = np.copy(rhop) * np.nan
for i in range(nz):
    rhop_int[:, i] = np.nansum(rhop[:, :i + 1] * dz * g, 1)
# calculate the pressure perturbation
rhop_int_bar = np.nansum(rhop_int * dz, 1) / nz / dz
pp = rhop_int - np.tile(rhop_int_bar, (nz, 1)).T
# ---------- calculate the energy flux ----------
fx = up_ni * pp
fy = vp_ni * pp
f = np.sqrt(fx ** 2 + fy ** 2)

np.savez(r'ReanaData/WOA23_new_horizontalEnergyFlux.npz', fx=fx, fy=fy)

# print(np.nanmax(f))
# plt.figure(1)
# plt.pcolormesh(mtime, depth, f.T, vmin=0, vmax=100)
# plt.colorbar()
# plt.show()
print('c')
