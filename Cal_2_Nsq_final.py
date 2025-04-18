import netCDF4 as nc
import numpy as np
import gsw
from datetime import datetime, timedelta
import scipy.interpolate as itp
import matplotlib.pyplot as plt


def cal_stratification(nt, nz, z, s, t, lat, lon):
    if nt == 1:
        s = s.reshape(1, -1)
        t = t.reshape(1, -1)
    ct = np.zeros((nt, nz))
    sig0 = np.zeros((nt, nz))
    N2 = np.zeros((nt, nz - 1))
    dtdz = np.zeros((nt, nz - 1))
    for m in range(nt):
        p = gsw.p_from_z(z, lat)
        SA = gsw.SA_from_SP(s[m, :], p, lon, lat)
        CT = gsw.CT_from_t(SA, t[m, :], p)
        [N2[m, :], p_mid] = gsw.Nsquared(SA, CT, p, lat)
        ct[m, :] = CT
        ze = gsw.z_from_p(p_mid.data, lat)
        sig0[m, :] = gsw.sigma0(SA, CT)
        dtdz[m, :] = np.diff(CT) / np.diff(z)
    N2[N2 < 0] = 1e-8
    return np.squeeze(ct), np.squeeze(sig0), np.squeeze(N2), np.squeeze(dtdz), ze


lat_moor = 36.23
lon_moor = -32.75
# ---------- using WOA23 data ----------
# monthly data
woa23 = np.load(r'ReanaData/WOA23_st.npz')
tempWOA = woa23['t']
saltWOA = woa23['s']
depthWOA = -woa23['z']
# ctWOA, sig0WOA, NsqWOA, dtdzWOA, zeWOA = (
#     cal_stratification(12, 57, depthWOA, saltWOA, tempWOA, lat_moor, lon_moor))
# np.savez(r'ReanaData/WOA23_stratification_tempByWOA.npz',
#          ct=ctWOA, sig0=sig0WOA, Nsq=NsqWOA, dtdz=dtdzWOA, ze=zeWOA)
# yearly data
# woa23Yearly = np.load(r'ReanaData/WOA23_st_yearly.npz')
# tempYearly = woa23Yearly['t']
# saltYearly = woa23Yearly['s']
# depthYearly = -woa23Yearly['z']
# nz = len(depthYearly)
# ctYearly, sig0Yearly, NsqYearly, dtdzYearly, zeYearly = (
#     cal_stratification(1, nz, depthYearly, saltYearly, tempYearly, lat_moor, lon_moor))
# plt.plot(NsqYearly, zeYearly)
# plt.show()
# np.savez(r'ReanaData/WOA23_stratification_tempByWOA_yearly.npz',
#          ct=ctYearly, sig0=sig0Yearly, Nsq=NsqYearly, dtdz=dtdzYearly, ze=zeYearly)
# ---------- using sensor data ----------
sensor = np.load(r'MoorData/SENSOR_temp.npz')
depthSensor = sensor['depth_sensor']
nz = len(depthSensor)
time = sensor['mtime_sensor']
dateSensor = [datetime(1, 1, 1) + timedelta(days=i - 367) for i in time]
nt = len(time)
tempSensor = sensor['temp']
saltGrid = np.empty((12, nz))
for i in range(12):
    itp_t = itp.interp1d(depthWOA, saltWOA[i, :], fill_value=np.nan, bounds_error=False)
    saltGrid[i, :] = itp_t(depthSensor)
saltSensor = np.empty((nt, nz))
for i in range(nt):
    m = dateSensor[i].month
    saltSensor[i, :] = saltGrid[m - 1, :]
ctSensor, sig0Sensor, NsqSensor, dtdzSensor, zeSensor = (
    cal_stratification(nt, nz, depthSensor, saltSensor, tempSensor, lat_moor, lon_moor))

plt.pcolormesh(range(6650), zeSensor, NsqSensor.T)
plt.colorbar()
plt.show()
# np.savez(r'ReanaData/WOA23_stratification_tempBySensor.npz',
#          ct=ctSensor, sig0=sig0Sensor, Nsq=NsqSensor, dtdz=dtdzSensor, ze=zeSensor)
print('c')
