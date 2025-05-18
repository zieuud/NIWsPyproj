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
# ---------- using WOA13 data ----------
# yearly data
woa23Yearly = np.load(r'../ReanaData/WOA13_st_yearly.npz')
tempYearly = woa23Yearly['t']
saltYearly = woa23Yearly['s']
depthYearly = -woa23Yearly['z']
nz = len(depthYearly)
ctYearly, sig0Yearly, NsqYearly, dtdzYearly, zeYearly = (
    cal_stratification(1, nz, depthYearly, saltYearly, tempYearly, lat_moor, lon_moor))
plt.figure(1, figsize=(4, 8))
plt.plot(NsqYearly, zeYearly)
plt.xlim([0, 1.5e-4])
plt.show()
# np.savez(r'../ReanaData/WOA13_stratification_tempByWOA_yearly.npz',
#          ct=ctYearly, sig0=sig0Yearly, Nsq=NsqYearly, dtdz=dtdzYearly, ze=zeYearly)
