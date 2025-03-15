import numpy as np
import gsw
import scipy.signal as sig
from matplotlib import pyplot as plt


def filter_ni(var, dt, nt, lat, c=1.25, N=4):
    fi = gsw.f(lat)/(2*np.pi)
    # Wn = np.array([(1. / c) * fi, c * fi]) * (2 * dt)
    Wn = np.array([0.9 * fi, 1.1 * fi]) * (2 * dt)
    b, a = sig.butter(N, Wn, btype='bandpass', output='ba')
    var_ni = np.copy(var) * np.nan
    for k in range(u.shape[-1]):
        tmp = np.copy(var[:, k])
        tmin = 0
        tmax = 1
        if tmp[~np.isnan(tmp)].shape[0] >= 10:  # filter only if there is at least 10 non-nan values
            while tmin < nt - 2:
                # - get intervals of non-nan data -
                while np.isnan(tmp[tmin]) and tmin < nt - 2:
                    tmin += 1
                    tmax = tmin + 1
                while ~np.isnan(tmp[tmax]) and tmax < nt - 1:
                    tmax += 1
                if tmax - tmin > 27:  # The length of the input vector x must be at least pad len, which is 27.
                    var_ni[tmin:tmax, k] = sig.filtfilt(b, a, tmp[tmin:tmax])
                tmin = tmax + 1
                tmax = tmin + 1
    return var_ni


def filter_lp(var, dt, nt, lat, c=1.25, N=4):
    fi = gsw.f(lat)/(2*np.pi)
    Wn = 0.5 * fi * (2 * dt)
    b, a = sig.butter(N, Wn, btype='lowpass', output='ba')
    var_lp = np.copy(var) * np.nan
    for k in range(u.shape[-1]):
        tmp = np.copy(var[:, k])
        tmin = 0
        tmax = 1
        if tmp[~np.isnan(tmp)].shape[0] >= 10:  # filter only if there is at least 10 non-nan values
            while tmin < nt - 2:
                # - get intervals of non-nan data -
                while np.isnan(tmp[tmin]) and tmin < nt - 2:
                    tmin += 1
                    tmax = tmin + 1
                while ~np.isnan(tmp[tmax]) and tmax < nt - 1:
                    tmax += 1
                if tmax - tmin > 27:  # The length of the input vector x must be at least pad len, which is 27.
                    var_lp[tmin:tmax, k] = sig.filtfilt(b, a, tmp[tmin:tmax], method='gust')
                tmin = tmax + 1
                tmax = tmin + 1
    return var_lp


moorData = np.load('ADCP_uv.npz')
u = np.transpose(moorData['u'])[:, :180]
v = np.transpose(moorData['v'])[:, :180]
depth = moorData['depth'][:180]
moorDate = moorData['mtime']
temp = np.load(r'ReanaData\WOA23_temp_grid.npy')[:, :180]
N2 = np.load(r'ReanaData\WOA23_N2_grid.npy')[:, :180]
sig0 = np.load(r'ReanaData\WOA23_sig0_grid.npy')[:, :180]
lat_moor = 36.23
lon_moor = -32.75
g = 9.81
nt = len(moorDate)
nz = len(depth)
dt = round((moorDate[1] - moorDate[0]) * 24 * 3600)
dz = depth[1] - depth[0]

# ---------- calculate the uv perturbation ----------
u_lp = filter_lp(u, dt, nt, lat_moor)
v_lp = filter_lp(v, dt, nt, lat_moor)
ubar0 = np.trapz((u - u_lp), -depth) / -depth[-1]
vbar0 = np.trapz((v - v_lp), -depth) / -depth[-1]
up = u - u_lp - np.tile(ubar0, (nz, 1)).T
vp = v - v_lp - np.tile(vbar0, (nz, 1)).T
up_ni = filter_ni(up, dt, nt, lat_moor)
vp_ni = filter_ni(vp, dt, nt, lat_moor)
# ---------- calculate the pressure perturbation ----------
# calculate the epsilon
temp_lp = filter_lp(temp, dt, nt, lat_moor)
tempp = temp - temp_lp
dtdz = np.diff(temp, axis=1) / np.diff(depth)
dtdz = np.concatenate((dtdz, dtdz[:, -1].reshape(-1, 1)), axis=1)
x = -tempp / dtdz
x_ni = filter_ni(x, dt, nt, lat_moor)
# calculate the rho prime
rhop = ((sig0 + 1000) / g) * N2 * x_ni
rhop_ni = filter_ni(rhop, dt, nt, lat_moor)
# calculate the p prime
p_ni = np.zeros((nt, nz))
for i in range(nz):
    p_ni[:, i] = np.trapz(rhop_ni[:, i:], -depth[i:]) * g
pbar_ni = -np.trapz(p_ni, -depth) / -depth[-1]
pp_ni = np.tile(pbar_ni, (nz, 1)).T + p_ni

fx = pp_ni * up_ni
fy = pp_ni * vp_ni
# ---------- save energy flux ----------
np.savez(r'ReanaData\WOA23_horizontalEnergyFlux.npz', fx=fx, fy=fy)

print('c')
