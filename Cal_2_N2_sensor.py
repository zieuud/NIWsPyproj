import numpy as np
import gsw
from datetime import datetime, timedelta
import scipy.interpolate as itp
import matplotlib.pyplot as plt
from astropy.time import Time

sensor = np.load(r'MoorData/SENSOR_temp.npz')
temp = sensor['temp']
z = sensor['depth_sensor']
mtime = sensor['mtime_sensor']
nt = len(mtime)
nz = len(z)
lat_moor = 36.23
lon_moor = -32.75

woa23 = np.load(r'ReanaData/WOA23_st.npz')
s = woa23['s']
depth = -woa23['z']
# ---------- unify the dimensions ----------
# create woa23 date
woa23Date = []
current_date = datetime(2015, 9, 15)
while current_date <= datetime(2016, 8, 15):
    woa23Date.append(current_date)
    month = current_date.month + 1 if current_date.month < 12 else 1
    year = current_date.year if month > 1 else current_date.year + 1
    current_date = datetime(year, month, 15)
woa23Date = Time(woa23Date).jd - Time(datetime(1, 1, 1)).jd + 366
# interpolate on moor date
s_hourly = np.zeros((nt, len(depth))) * np.nan
for i in range(len(depth)):
    itp_t = itp.interp1d(woa23Date, s[:, i])
    s_hourly[:, i] = itp_t(mtime)
# interpolate on sensor depth
s_hourly_sensor = np.zeros((nt, nz))
for i in range(nt):
    itp_t = itp.interp1d(depth, s_hourly[i, :], fill_value=np.nan, bounds_error=False)
    s_hourly_sensor[i, :] = itp_t(z)

# ---------- calculate N2 sig0 dtdz ----------
s = s_hourly_sensor
t = temp
sig0 = np.zeros((nt, nz))
N2 = np.zeros((nt, nz - 1))
dtdz = np.zeros((nt, nz - 1))
for m in range(nt):
    p = gsw.p_from_z(z, lat_moor)
    SA = gsw.SA_from_SP(s[m - 1, :], p, lon_moor, lat_moor)
    CT = gsw.CT_from_t(SA, t[m - 1, :], p)
    [N2[m - 1, :], p_mid] = gsw.Nsquared(SA, CT, p, lat_moor)
    ze = gsw.z_from_p(p_mid.data, lat_moor)
    sig0[m - 1, :] = gsw.sigma0(SA, CT)
    dtdz[m - 1, :] = np.diff(CT) / np.diff(z)
N2[N2 < 0] = 1e-8
np.save(r'ReanaData/WOA23_N2_sensorGrid.npy', N2)
# ---------- interpolate on adcp depth ----------
adcp = np.load(r'MoorData/ADCP_uv.npz')
z_adcp = adcp['depth_adcp']
nz_adcp = len(z_adcp)

t_moor = np.zeros((nt, nz_adcp))
sig0_moor = np.zeros((nt, nz_adcp))
N2_moor = np.zeros((nt, nz_adcp))
dtdz_moor = np.zeros((nt, nz_adcp))
for i in range(nt):
    itp_t = itp.interp1d(z, t[i, :], fill_value=np.nan, bounds_error=False)
    t_moor[i, :] = itp_t(z_adcp)
    itp_t = itp.interp1d(z, sig0[i, :], fill_value=np.nan, bounds_error=False)
    sig0_moor[i, :] = itp_t(z_adcp)
    itp_t = itp.interp1d(ze, N2[i, :], fill_value=np.nan, bounds_error=False)
    N2_moor[i, :] = itp_t(z_adcp)
    itp_t = itp.interp1d(ze, dtdz[i, :], fill_value=np.nan, bounds_error=False)
    dtdz_moor[i, :] = itp_t(z_adcp)

np.save(r'ReanaData\WOA23_N2_tsensor_grid.npy', N2_moor)
np.save(r'ReanaData\WOA23_dtdz_tsensor_grid.npy', dtdz_moor)
np.save(r'ReanaData\WOA23_sig0_tsensor_grid.npy', sig0_moor)
np.save(r'ReanaData\WOA23_temp_tsensor_grid.npy', t_moor)
print('c')
