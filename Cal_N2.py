import numpy as np
import gsw
from datetime import datetime, timedelta
import scipy.interpolate as itp
import matplotlib.pyplot as plt
from astropy.time import Time

woa23 = np.load(r'ReanaData\WOA23_st.npz')
s = woa23['s']
t = woa23['t']
z = -woa23['z']
[lat, lon] = woa23['loc']
sig0 = np.zeros((12, 57))
N2 = np.zeros((12, 56))
for m in range(1, 13):
    p = gsw.p_from_z(z, lat)
    SA = gsw.SA_from_SP(s[m - 1, :], p, lon, lat)
    CT = gsw.CT_from_t(SA, t[m - 1, :], p)
    [N2[m - 1, :], p_mid] = gsw.Nsquared(SA, CT, p, lat)
    ze = gsw.z_from_p(p_mid.data, lat)
    sig0[m - 1, :] = gsw.sigma0(SA, CT)

# ----------interpolate----------
# load moor date and depth
moorData = np.load('ADCP_uv.npz')
moorDate = moorData['mtime']
moorDepth = moorData['depth']
# create woa23 date
woa23Date = []
current_date = datetime(2015, 9, 15)
while current_date <= datetime(2016, 8, 15):
    woa23Date.append(current_date)
    month = current_date.month + 1 if current_date.month < 12 else 1
    year = current_date.year if month > 1 else current_date.year + 1
    current_date = datetime(year, month, 15)
# woa23Date = [date.toordinal() for date in woa23Date]
woa23Date = Time(woa23Date).jd - 1721424.5 + 366
# interpolate on time
N2_woa_hourly = np.zeros((len(moorDate), len(z)-1))
sig0_woa_hourly = np.zeros((len(moorDate), len(z)))
temp_woa_hourly = np.zeros((len(moorDate), len(z)))
for k in range(len(z)-1):
    itp_t = itp.interp1d(woa23Date, N2[:, k])
    N2_woa_hourly[:, k] = itp_t(moorDate)
    itp_t = itp.interp1d(woa23Date, sig0[:, k])
    sig0_woa_hourly[:, k] = itp_t(moorDate)
    itp_t = itp.interp1d(woa23Date, t[:, k])
    temp_woa_hourly[:, k] = itp_t(moorDate)
# interpolate on depth
N2_woa_grid = np.zeros((len(moorDate), len(moorDepth)-1))
sig0_woa_grid = np.zeros((len(moorDate), len(moorDepth)))
temp_woa_grid = np.zeros((len(moorDate), len(moorDepth)))
for j in range(len(moorDate)):
    itp_t = itp.interp1d(ze, N2_woa_hourly[j, :], fill_value=np.nan, bounds_error=False)
    N2_woa_grid[j, :] = itp_t(moorDepth[1:])
    itp_t = itp.interp1d(z, sig0_woa_hourly[j, :], fill_value=np.nan, bounds_error=False)
    sig0_woa_grid[j, :] = itp_t(moorDepth)
    itp_t = itp.interp1d(z, temp_woa_hourly[j, :], fill_value=np.nan, bounds_error=False)
    temp_woa_grid[j, :] = itp_t(moorDepth)

np.save(r'ReanaData\WOA23_N2_grid.npy', N2_woa_grid)
np.save(r'ReanaData\WOA23_sig0_grid.npy', sig0_woa_grid)
np.save(r'ReanaData\WOA23_temp_grid.npy', temp_woa_grid)


