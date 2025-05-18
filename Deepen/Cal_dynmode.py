import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as itp
from datetime import datetime, timedelta


def dynmodes(N2, z, nmodes):
    rho0 = 1025.
    reverse = False
    # --- ensure descending profiles
    if z[0] < z[-1]:
        reverse = True
        N2 = N2[::-1]
        z = z[::-1]
    # --- check surface value
    if z[0] > 0:
        z = np.concatenate(([0], z))
        N2 = np.concatenate((N2[0], N2))
    nz = z.shape[0]
    # --- calculate depth and spacing
    dz = z[:-1] - z[1:]  # spacing
    zm = z[:-1] - 0.5 * dz  # mid-point depth
    dzm = np.zeros(nz)  # mid-point spacing
    dzm[1:-1] = zm[:nz - 2] - zm[1:nz - 1]
    dzm[0] = dzm[1]
    dzm[-1] = dzm[-2]
    # --- set matrices
    A = np.zeros((nz, nz))
    B = np.zeros((nz, nz))
    for i in np.arange(1, nz - 1):
        A[i, i] = 1. / (dz[i - 1] * dzm[i]) + 1. / (dz[i] * dzm[i])
        A[i, i - 1] = -1. / (dz[i - 1] * dzm[i])
        A[i, i + 1] = -1. / (dz[i] * dzm[i])
    for i in np.arange(nz):
        B[i, i] = N2[i]
    A[0, 0] = -1.
    A[nz - 1, 0] = -1.
    # --- extract eigen stuff (w-modes)
    e, wmodes = np.linalg.eig(np.dot(np.linalg.inv(B), A))

    ind = np.argwhere(np.imag(e) == 0)  # only Real eigenvalues (physical meaning)
    e = np.squeeze(e[ind])
    wmodes = np.squeeze(wmodes[:, ind])

    ind = np.argwhere(e > 1e-10)  # no zeros eigenvalues
    e = np.squeeze(e[ind])
    wmodes = np.squeeze(wmodes[:, ind])

    ind = np.argsort(e)  # sort eigenvalues and corresponding eigenvectors
    e = np.squeeze(e[ind])
    wmodes = np.squeeze(wmodes[:, ind])
    # --- create u,v,p-modes (first derivative of w-modes)
    nm = e.shape[0]  # number of relevant modes
    ce = 1. / np.sqrt(e)
    pmodes = np.zeros(wmodes.shape)
    for i in np.arange(nm):
        pr = np.diff(wmodes[:, i]) / dz
        pr = pr * rho0 * ce[i] * ce[i]  # for homogeneity
        pmodes[1:nz - 1, i] = .5 * (pr[1:] + pr[:-1])
        pmodes[0, i] = pr[0]
        pmodes[-1, i] = pr[-1]
    # --- adds the barotropic mode
    bt_mode = np.ones((nz, 1))
    wmodes = np.concatenate((bt_mode, wmodes), axis=1)
    pmodes = np.concatenate((bt_mode, pmodes), axis=1)
    ce = np.concatenate(([np.nan], ce))
    # --- select first nmodes
    wmodes = wmodes[:, :nmodes].T
    pmodes = pmodes[:, :nmodes].T
    ce = ce[:nmodes]
    if reverse:
        wmodes = wmodes[:, ::-1]
        pmodes = pmodes[:, ::-1]
    return wmodes, pmodes, ce


nmodes = 11
adcp = np.load(r'../MoorData/ADCP_uv.npz')
depthMoor = adcp['depth_adcp']
nzMoor = len(depthMoor)
timeMoor = adcp['mtime_adcp']
ntMoor = len(timeMoor)
dateMoor = [datetime(1, 1, 1) + timedelta(days=m - 367) for m in timeMoor]

# ---------- calculate pmodes&ce on moor grid with WOA23 yearly temperature data ----------
stratification = np.load(r'../ReanaData/WOA23_stratification_tempByWOA_yearly.npz')
Nsq = stratification['Nsq']
ze = stratification['ze']

upper = np.array([-4, -8, -12, -20])
depthMoor_full = np.concatenate([upper, depthMoor])
itp_t = itp.interp1d(ze, Nsq, fill_value=1e-8, bounds_error=False, kind='cubic')
Nsq_interp = itp_t(depthMoor_full)
plt.figure(1, figsize=(4, 8))
plt.plot(Nsq_interp, depthMoor_full)
plt.xlim([0, 1.5e-4])
plt.show()

wmodes, pmodes, ce = dynmodes(Nsq_interp, depthMoor_full, nmodes)  # 取 4:
# wmodes, pmodes, ce = dynmodes(Nsq, ze, nmodes)
# pmodesMoor = np.zeros((nmodes, nzMoor)) * np.nan
# for m in range(nmodes):
#     itp_t = itp.interp1d(ze, pmodes[m, :], fill_value=np.nan, bounds_error=False, kind='cubic')
#     pmodesMoor[m, :] = itp_t(depthMoor)
# np.savez(r'../ReanaData/WOA13_pmodes_moorGrid_yearly.npz', pmodes=pmodesMoor, Nsq=Nsq, ze=ze)
np.savez(r'../ReanaData/WOA23_pmodes_moorGrid_yearly.npz', pmodes=pmodes, Nsq=Nsq, ze=ze)
# ---------- calculate pmodes&ce on moor grid with WOA23 monthly + seasonal data ----------

