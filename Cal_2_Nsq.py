import netCDF4 as nc
import numpy as np
import gsw
from datetime import datetime, timedelta
import scipy.interpolate as itp
import matplotlib.pyplot as plt
from astropy.time import Time


def cal_stratification(nt, nz, z, s, t, lat_moor):
    ct = np.zeros((nt, nz))
    sig0 = np.zeros((nt, nz))
    N2 = np.zeros((nt, nz - 1))
    dtdz = np.zeros((nt, nz - 1))
    for m in range(nt):
        p = gsw.p_from_z(z, lat_moor)
        SA = gsw.SA_from_SP(s[m, :], p, lon_moor, lat_moor)
        CT = gsw.CT_from_t(SA, t[m, :], p)
        [N2[m, :], p_mid] = gsw.Nsquared(SA, CT, p, lat_moor)
        ct[m, :] = CT
        ze = gsw.z_from_p(p_mid.data, lat_moor)
        sig0[m, :] = gsw.sigma0(SA, CT)
        dtdz[m, :] = np.diff(CT) / np.diff(z)
    N2[N2 < 0] = 1e-8
    return ct, sig0, N2, dtdz, ze


lat_moor = 36.23
lon_moor = -32.75
# ---------- calculate Nsq by sensor temperature ----------
# read sensor data
sensor = np.load(r'MoorData/SENSOR_temp.npz')
temp = sensor['temp']
z = sensor['depth_sensor']
mtime = sensor['mtime_sensor']
nt = len(mtime)
nz_sensor = len(z)
# read woa23 data
woa23 = np.load(r'ReanaData/WOA23_st.npz')
t = woa23['t']
s = woa23['s']
depth = -woa23['z']
nz_woa23 = len(depth)
# interpolate salinity data from woa23 on sensor data and sensor depth
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
t_hourly = np.zeros((nt, nz_woa23)) * np.nan
s_hourly = np.zeros((nt, nz_woa23)) * np.nan
for i in range(nz_woa23):
    itp_t = itp.interp1d(woa23Date, t[:, i])
    t_hourly[:, i] = itp_t(mtime)
    itp_t = itp.interp1d(woa23Date, s[:, i])
    s_hourly[:, i] = itp_t(mtime)
# interpolate on sensor depth
s_hourly_sensor = np.zeros((nt, nz_sensor))
for i in range(nt):
    itp_t = itp.interp1d(depth, s_hourly[i, :], fill_value=np.nan, bounds_error=False)
    s_hourly_sensor[i, :] = itp_t(z)
# calculate N2 sig0 dtdz
ct_sensor, sig0_sensor, Nsq_sensor, dtdz_sensor, ze_sensor = cal_stratification(nt, nz_sensor, z, s_hourly_sensor, temp, lat_moor)

# ---------- calculate Nsq by woa23 temperature ----------
ct_woa, sig0_woa, Nsq_woa, dtdz_woa, ze_woa = cal_stratification(nt, nz_woa23, depth, s_hourly, t_hourly, lat_moor)

# ---------- calculate Nsq by woa23&sensor fusion data ----------
fusion = np.load(r'ReanaData/WOA23_st_fusion.npz')
tempFusion = fusion['tempFusion']
saltFusion = fusion['saltFusion']
depthFusion = fusion['depthFusion']
ct_fusion, sig0_fusion, Nsq_fusion, dtdz_fusion, ze_fusion = (
    cal_stratification(nt, len(depthFusion), depthFusion, saltFusion, tempFusion, lat_moor))
np.savez(r'ReanaData/WOA23_stratification.npz', ct_fuison=ct_fusion, sig0_fusion=sig0_fusion,
        Nsq_fusion=Nsq_fusion, dtdz_fusion=dtdz_fusion, depth_fusion=depthFusion, ze_fusion=ze_fusion)

# ---------- interpolate on adcp depth ----------
adcp = np.load(r'MoorData/ADCP_uv.npz')
depthMoor = adcp['depth_adcp']
nzMoor = len(depthMoor)
ctMoor = np.zeros((nt, nzMoor))
sig0Moor = np.zeros((nt, nzMoor))
NsqMoor = np.zeros((nt, nzMoor))
dtdzMoor = np.zeros((nt, nzMoor))
for i in range(nt):
    itp_t = itp.interp1d(depthFusion, ct_fusion[i, :], fill_value=np.nan, bounds_error=False)
    ctMoor[i, :] = itp_t(depthMoor)
    itp_t = itp.interp1d(depthFusion, sig0_fusion[i, :], fill_value=np.nan, bounds_error=False)
    sig0Moor[i, :] = itp_t(depthMoor)
    itp_t = itp.interp1d(ze_fusion, Nsq_fusion[i, :], fill_value=np.nan, bounds_error=False)
    NsqMoor[i, :] = itp_t(depthMoor)
    itp_t = itp.interp1d(ze_fusion, dtdz_fusion[i, :], fill_value=np.nan, bounds_error=False)
    dtdzMoor[i, :] = itp_t(depthMoor)

np.savez(r'ReanaData/WOA23_stratification_adcpGrid.npz', ct_fuison_adcpGrid=ctMoor,
         sig0_fusion_adcpGrid=sig0Moor, Nsq_fusion_adcpGrid=NsqMoor, dtdz_fusion_adcpGrid=dtdzMoor)

# plt.plot(np.nanmean(NsqMoor, 0), depthMoor)
# plt.show()
# for checking
plt.figure(1, figsize=(8, 8))
plt.subplot(1, 3, 1)
plt.plot(np.nanmean(Nsq_sensor, 0), ze_sensor)
plt.ylim([-2000, 0])
plt.subplot(1, 3, 2)
plt.plot(np.nanmean(Nsq_woa, 0), ze_woa)
plt.ylim([-2000, 0])
plt.subplot(1, 3, 3)
plt.plot(np.nanmean(Nsq_fusion, 0), ze_fusion)
plt.ylim([-2000, 0])
# plt.savefig(r'figures/compare_Nsq_mean.jpg', dpi=300)

plt.figure(2, figsize=(15, 6))
plt.subplot(3, 1, 1)
plt.pcolormesh(mtime, ze_sensor, Nsq_sensor.T, vmin=0, vmax=1e-4)
plt.colorbar()
plt.ylim([-2000, 0])
plt.subplot(3, 1, 2)
plt.pcolormesh(mtime, ze_woa, Nsq_woa.T, vmin=0, vmax=1e-4)
plt.colorbar()
plt.ylim([-2000, 0])
plt.subplot(3, 1, 3)
plt.pcolormesh(mtime, ze_fusion, Nsq_fusion.T, vmin=0, vmax=1e-4)
plt.colorbar()
plt.ylim([-2000, 0])
# plt.savefig(r'figures/compare_Nsq.jpg', dpi=300)
# plt.show()
