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


stratification = np.load(r'ReanaData/WOA23_stratification_tempByWOA.npz')
Nsq = stratification['Nsq']
ze = stratification['ze']

# ---------- plot the mean vertical structure of modes ----------
nmodes = 11
Nmean = np.nanmean(Nsq, 0)
wmodes, pmodes, ce = dynmodes(np.squeeze(Nmean), ze, nmodes)
adcp = np.load(r'MoorData/ADCP_uv.npz')
depthMoor = adcp['depth_adcp']
nzMoor = len(depthMoor)
timeMoor = adcp['mtime_adcp']
ntMoor = len(timeMoor)
dateMoor = [datetime(1, 1, 1) + timedelta(days=m - 367) for m in timeMoor]
pmodes_mean = np.zeros((nmodes, nzMoor)) * np.nan
for m in range(nmodes):
    itp_t = itp.interp1d(ze, pmodes[m, :], fill_value=np.nan, bounds_error=False, kind='cubic')
    pmodes_mean[m, :] = itp_t(depthMoor)
np.savez(r'ReanaData/WOA23_pmodes_mean.npz', pmodes=pmodes_mean, Nsq=Nmean, ze=ze)

plt.figure(1, figsize=(10, 12))
for i in range(11):
    if i >= 6:
        j = i + 2
    else:
        j = i + 1
    plt.subplot(2, 6, j)
    if i == 0:
        plt.plot(Nmean, ze)
        plt.ylabel('Depth (m)')
        plt.title(r'$N^{2}$')
        plt.ylim([-2000, 0])
    else:
        if pmodes_mean[i, 0] < 0:
            pmodes_mean[i, :] = -pmodes_mean[i, :]
        plt.plot(pmodes_mean[i, :].T, depthMoor)
        if i == 6:
            plt.ylabel('Depth (m)')
        else:
            plt.yticks([])
        plt.xticks([])
        plt.plot([0, 0], [0, -2000], 'k--')
        plt.title('mode {}'.format(i))
        plt.ylim([-2000, 0])
# plt.savefig(r'figures/vertical_modes_structure.jpg', dpi=300)
# plt.show()

# ---------- calculate pmodes&ce on moor grid ----------
pmodesMoor = np.zeros((ntMoor, nmodes, nzMoor)) * np.nan
ceMoor = np.zeros((ntMoor, nmodes)) * np.nan
for t in range(ntMoor):
    t2 = dateMoor[t].month
    wmodes, pmodes, ce = dynmodes(Nsq[t2 - 1, :], ze, nmodes)
    for m in range(nmodes):
        itp_t = itp.interp1d(ze, pmodes[m, :], fill_value=np.nan, bounds_error=False, kind='cubic')
        pmodesMoor[t, m, :] = itp_t(depthMoor)
    ceMoor[t, :] = ce
pmodesMoor[:, 0, :] = 1.  # 正压模态统一为1. 避免插值导致偏离
pmodesMoor[pmodesMoor[:, :, 0] < 0] = -pmodesMoor[pmodesMoor[:, :, 0] < 0]  # 统一模态结构方向
np.savez(r'ReanaData/WOA23_pmodes_moorGrid.npz', pmodes=pmodesMoor, ce=ceMoor)
