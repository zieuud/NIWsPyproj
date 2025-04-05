import numpy as np
import gsw
from datetime import datetime, timedelta
import scipy.interpolate as itp
import matplotlib.pyplot as plt
from astropy.time import Time

woa23 = np.load(r'WOA23_st.npz')
s = woa23['s']
t = woa23['t']
z = -woa23['z']
[lat, lon] = woa23['loc']

sig0 = np.zeros((27, 57))
N2 = np.zeros((27, 56))
dtdz = np.zeros((27, 56))
for m in range(27):
    p = gsw.p_from_z(z, lat)
    SA = gsw.SA_from_SP(s[m, :], p, lon, lat)
    CT = gsw.CT_from_t(SA, t[m, :], p)
    [N2month, p_mid] = gsw.Nsquared(SA, CT, p, lat)
    ze = gsw.z_from_p(p_mid.data, lat)
    N2month[N2month < 0] = 1e-8
    N2[m, :] = N2month
    sig0[m, :] = gsw.sigma0(SA, CT)
    dtdz[m, :] = np.diff(CT) / np.diff(z)
# ----------interpolate----------
# load moor date and depth
moorData = np.load('mooringICE.npz')
moorDate = moorData['mtime'][:, 0]
moorDepth = moorData['depthCurr']
# create woa23 date
woa23Date = []
current_date = datetime(2015, 6, 15)
while current_date <= datetime(2017, 8, 15):
    woa23Date.append(current_date)
    month = current_date.month + 1 if current_date.month < 12 else 1
    year = current_date.year if month > 1 else current_date.year + 1
    current_date = datetime(year, month, 15)
woa23Date = [(i - datetime(1950, 1, 1)).days for i in woa23Date]
# interpolate on time
N2_woa_hourly = np.zeros((len(moorDate), len(z)-1))
dtdz_woa_hourly = np.zeros((len(moorDate), len(z)-1))
sig0_woa_hourly = np.zeros((len(moorDate), len(z)))
temp_woa_hourly = np.zeros((len(moorDate), len(z)))
for k in range(len(z)-1):
    itp_t = itp.interp1d(woa23Date, N2[:, k])
    N2_woa_hourly[:, k] = itp_t(moorDate)
    itp_t = itp.interp1d(woa23Date, dtdz[:, k])
    dtdz_woa_hourly[:, k] = itp_t(moorDate)
for m in range(len(z)):
    itp_t = itp.interp1d(woa23Date, sig0[:, m])
    sig0_woa_hourly[:, m] = itp_t(moorDate)
    itp_t = itp.interp1d(woa23Date, t[:, m])
    temp_woa_hourly[:, m] = itp_t(moorDate)
# interpolate on depth
N2_woa_grid = np.zeros((len(moorDate), len(moorDepth)))
dtdz_woa_grid = np.zeros((len(moorDate), len(moorDepth)))
sig0_woa_grid = np.zeros((len(moorDate), len(moorDepth)))
temp_woa_grid = np.zeros((len(moorDate), len(moorDepth)))
for j in range(len(moorDate)):
    itp_t = itp.interp1d(ze, N2_woa_hourly[j, :], fill_value=np.nan, bounds_error=False)
    N2_woa_grid[j, :] = itp_t(moorDepth)
    itp_t = itp.interp1d(ze, dtdz_woa_hourly[j, :], fill_value=np.nan, bounds_error=False)
    dtdz_woa_grid[j, :] = itp_t(moorDepth)
    itp_t = itp.interp1d(z, sig0_woa_hourly[j, :], fill_value=np.nan, bounds_error=False)
    sig0_woa_grid[j, :] = itp_t(moorDepth)
    itp_t = itp.interp1d(z, temp_woa_hourly[j, :], fill_value=np.nan, bounds_error=False)
    temp_woa_grid[j, :] = itp_t(moorDepth)

print('c')

np.save(r'WOA23_N2_grid.npy', N2_woa_grid)
np.save(r'WOA23_sig0_grid.npy', sig0_woa_grid)
np.save(r'WOA23_temp_grid.npy', temp_woa_grid)
np.save(r'WOA23_dtdz_grid.npy', dtdz_woa_grid)


