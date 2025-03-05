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
u = np.transpose(moorData['u'])
v = np.transpose(moorData['v'])  # transpose to (time, depth)
depth = moorData['depth']
moorDate = moorData['mtime']
lat = 36.23
nt = len(moorDate)
dt = round((moorDate[1] - moorDate[0]) * 24 * 3600)

u_ni = filter_ni(u - filter_lp(u, dt, nt, lat), dt, nt, lat)
v_ni = filter_ni(u - filter_lp(v, dt, nt, lat), dt, nt, lat)
KE_ni = 1 / 2 * 1025 * (u_ni ** 2 + v_ni ** 2)

np.savez('ADCP_uv_ni.npz', u_ni=u_ni, v_ni=v_ni, KE_ni=KE_ni)

# [depth_axis, time_axis] = np.meshgrid(depth, moorDate)
# plt.pcolor(time_axis, depth_axis, KE_ni, cmap='Oranges')
# plt.colorbar()
# plt.clim(0, 10)
# plt.show()
