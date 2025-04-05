import matplotlib.pyplot as plt
import numpy as np
import func_0_filter as func

# read adcp data
adcp = np.load(r'mooringICE.npz')
depth = adcp['depthCurr']
mtime = adcp['mtime'][:, 0]
lat_moor = adcp['latitude']
lon_moor = adcp['longitude']
u = adcp['u']
v = adcp['v']
# constant parameter
dt = 3600
nt = len(mtime)
dz = depth[1] - depth[0]
nz = len(depth)
g = 9.81
# read woa23 data
temp = np.load(r'WOA23_temp_grid.npy')
Nsq = np.load(r'WOA23_N2_grid.npy')
sig0 = np.load(r'WOA23_sig0_grid.npy')
dtdz = np.load(r'WOA23_dtdz_grid.npy')

# ------------ processing ------------
print(' ... processing ... ')
print('######################## 1. COMPUTE VELOCITY PERTURBATION ##############################')
u_lp = func.filter_lp(u, dt, nt, lat_moor)
v_lp = func.filter_lp(v, dt, nt, lat_moor)
up = u - u_lp
vp = v - v_lp
up = up - np.transpose(np.tile(np.nanmean(up * dz, axis=-1) / depth[-1], (nz, 1)))
vp = vp - np.transpose(np.tile(np.nanmean(vp * dz, axis=-1) / depth[-1], (nz, 1)))
# --- filtering ---
u_ni = func.filter_ni(up, dt, nt, lat_moor)
v_ni = func.filter_ni(vp, dt, nt, lat_moor)
print('######################## 2. COMPUTE PRESSURE PERTURBATION ##############################')
t_lp = func.filter_vlp(temp, dt, nt, lat_moor)
tp = temp - t_lp
x = -tp / dtdz

# --- filtering ---
# x_ni = func.filter_ni(x, dt, nt, lat_moor)

# --- compute pressure anomaly ---
rhoa = ((sig0 + 1000) / g) * Nsq * x
# rhoa_ni = ((sig0 + 1000) / g) * Nsq * x_ni

p = np.zeros((nt, nz))
# p_ni = np.zeros((nt, nz))

for k in range(nz):
    p[:, k] = np.nansum(rhoa[:, k:] * dz * g, axis=1)
    # p_ni[:, k] = np.nansum(rhoa_ni[:, k:] * dz * g, axis=1)

# --- baroclinicity condition ---
pbar = np.transpose(np.tile(np.nansum(p, axis=-1) / depth[-1], (nz, 1)))
# pbar_ni = np.transpose(np.tile(np.nansum(p_ni, axis=-1) / depth[-1], (1, 1)))
p = p - pbar
# p_ni = p_ni - pbar_ni

fx = u_ni * p
fy = v_ni * p
print(np.nanmax(fx))
print(np.nanmax(fy))
print('c')
