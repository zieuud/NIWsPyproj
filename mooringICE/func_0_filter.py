import numpy as np
import gsw
import scipy.signal as sig


def filter_ni(var, dt, nt, lat, N=4):
    fi = gsw.f(lat)/(2*np.pi)
    # Wn = np.array([(1. / c) * fi, c * fi]) * (2 * dt)
    Wn = np.array([0.8 * fi, 1.2 * fi]) * (2 * dt)
    b, a = sig.butter(N, Wn, btype='bandpass', output='ba')
    var_ni = np.copy(var) * np.nan
    for k in range(var.shape[-1]):
        tmp = np.copy(var[:, k])
        tmin = 0
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
    var_ni = np.squeeze(var_ni)
    return var_ni


def filter_lp(var, dt, nt, lat, N=4):
    fi = gsw.f(lat)/(2*np.pi)
    Wn = 0.5 * fi * (2 * dt)
    b, a = sig.butter(N, Wn, btype='lowpass', output='ba')
    var_lp = np.copy(var) * np.nan
    for k in range(var.shape[-1]):
        tmp = np.copy(var[:, k])
        tmin = 0
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
    var_lp = np.squeeze(var_lp)
    return var_lp


def filter_vlp(var, dt, nt, lat, N=4):
    fi = gsw.f(lat)/(2*np.pi)
    Wn = 0.1 * fi * (2 * dt)
    b, a = sig.butter(N, Wn, btype='lowpass', output='ba')
    var_vlp = np.copy(var) * np.nan
    for k in range(var.shape[-1]):
        tmp = np.copy(var[:, k])
        tmin = 0
        if tmp[~np.isnan(tmp)].shape[0] >= 10:  # filter only if there is at least 10 non-nan values
            while tmin < nt - 2:
                # - get intervals of non-nan data -
                while np.isnan(tmp[tmin]) and tmin < nt - 2:
                    tmin += 1
                tmax = tmin + 1
                while ~np.isnan(tmp[tmax]) and tmax < nt - 1:
                    tmax += 1
                if tmax - tmin > 27:  # The length of the input vector x must be at least pad len, which is 27.
                    var_vlp[tmin:tmax, k] = sig.filtfilt(b, a, tmp[tmin:tmax], method='gust')
                tmin = tmax + 1
    var_vlp = np.squeeze(var_vlp)
    return var_vlp