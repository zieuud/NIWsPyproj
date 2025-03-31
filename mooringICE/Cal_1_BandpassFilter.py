import numpy as np
import gsw
import scipy.signal as sig
from matplotlib import pyplot as plt


def filter_ni(var, dt, nt, lat, N=4):
    fi = gsw.f(lat) / (2 * np.pi)
    print(fi)
    Wn = np.array([0.8 * fi, 1.2 * fi]) * 2 * dt
    b, a = sig.butter(N, Wn, btype='band', output='ba')
    var_ni = np.copy(var) * np.nan
    for k in range(var.shape[-1]):
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
                    # print(tmin, tmax, k)
                tmin = tmax + 1
                tmax = tmin + 1
        # else:
            # print(k)
    return var_ni


def filter_lp(var, dt, nt, lat, N=4):
    fi = gsw.f(lat) / (2 * np.pi)
    Wn = 0.5 * fi * (2 * dt)
    b, a = sig.butter(N, Wn, btype='lowpass', output='ba')
    var_lp = np.copy(var) * np.nan
    for k in range(var.shape[-1]):
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


ice = np.load(r'mooringICE.npz')
latitude = ice['latitude']
depth = ice['depthCurr']
mtime = ice['mtime']
u = ice['u']
v = ice['v']
ke = ice['ke']
dt = 3600
nt = len(mtime)

u_ni = filter_ni(u, dt, nt, latitude)
v_ni = filter_ni(v, dt, nt, latitude)
ke_ni = 1 / 2 * 1025 * (u_ni ** 2 + v_ni ** 2)
np.savez(r'mooring_uv_ni.npz', u_ni=u_ni, v_ni=v_ni, ke_ni=ke_ni)


plt.figure(1)
plt.pcolormesh(mtime, depth, ke_ni.T, cmap='YlOrBr', vmin=0, vmax=6)
plt.colorbar()
plt.show()
