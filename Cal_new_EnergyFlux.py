import numpy as np
import func_0_filter as func

adcp = np.load(r'MoorData\ADCP_uv.npz')
depth = adcp['depth_adcp']
mtime = adcp['mtime_adcp']
lat_moor = 36.23
lon_moor = -32.75
u = adcp['u'].T
v = adcp['v'].T
temp = np.load(r'MoorData\SENSOR_temp_interpolate.npy').T
dt = 3600
nt = len(mtime)
dz = depth[1] - depth[0]
nz = len(depth)
g = 9.81

# ---------- calculate the uv perturbation ----------
u_lp = func.filter_lp(u, dt, nt, lat_moor)
v_lp = func.filter_lp(v, dt, nt, lat_moor)
ubar0 = np.trapz((u - u_lp), -depth) / -depth[-1]
vbar0 = np.trapz((v - v_lp), -depth) / -depth[-1]
up = u - u_lp - np.tile(ubar0, (nz, 1)).T
vp = v - v_lp - np.tile(vbar0, (nz, 1)).T
up_ni = func.filter_ni(up, dt, nt, lat_moor)
vp_ni = func.filter_ni(vp, dt, nt, lat_moor)
# ---------- calculate the pressure perturbation ----------
# calculate the epsilon
temp_lp = func.filter_lp(temp, dt, nt, lat_moor)
tempp = temp - temp_lp

print('c')
