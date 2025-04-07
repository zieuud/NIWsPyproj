from datetime import datetime, timedelta
import numpy as np
from matplotlib import pyplot as plt
import scipy.interpolate as itp


Nsq = np.load(r'ReanaData/WOA23_stratification_tempByWOA.npz')['Nsq']
ze = np.load(r'ReanaData/WOA23_stratification_tempByWOA.npz')['ze']
moorData = np.load('MoorData/ADCP_uv.npz')
depthMoor = moorData['depth_adcp']
timeMoor = moorData['mtime_adcp']
nt = len(timeMoor)
nz = len(depthMoor)
dateMoor = [datetime(1, 1, 1) + timedelta(days=m-367) for m in timeMoor]
NsqMoor = np.empty((nt, nz))
for i in range(nt):
    m = dateMoor[i].month
    itp_t = itp.interp1d(ze, Nsq[m - 1, :], fill_value=np.nan, bounds_error=False)
    NsqMoor[i, :] = itp_t(depthMoor)
maxIdx = 181

uv_ni = np.load('MoorData/ADCP_uv_ni_ByYu.npz')
u_ni = uv_ni['u_ni']
v_ni = uv_ni['v_ni']
KE_ni = uv_ni['ke_ni']
u_ni_wkb = np.copy(u_ni)
v_ni_wkb = np.copy(v_ni)
N2_averaged = np.nanmean(NsqMoor, 1)  # 最后一个N2非nan位置
for i in range(nt):
    u_ni_wkb[i, :maxIdx] = u_ni[i, :maxIdx] * np.sqrt(np.sqrt(N2_averaged[i])/np.sqrt(NsqMoor[i, :maxIdx]))
    v_ni_wkb[i, :maxIdx] = v_ni[i, :maxIdx] * np.sqrt(np.sqrt(N2_averaged[i])/np.sqrt(NsqMoor[i, :maxIdx]))
KE_ni_wkb = 1/2*1025*(u_ni_wkb**2+v_ni_wkb**2)

# monthly compose
month_last = 8
month_now = 8
idx_start = 0
idx_end = idx_start
KE_ni_monthly = np.zeros((12, 245))
KE_ni_wkb_monthly = np.zeros((12, 245))
for i in range(nt):
    month_now = dateMoor[i].month
    if month_now == month_last and idx_end != 6650-2:
        idx_end = i
    else:
        print(month_last, idx_start, idx_end)
        KE_ni_monthly[month_last-1, :] = np.nanmean(KE_ni[idx_start:idx_end, :], 0)
        KE_ni_wkb_monthly[month_last-1, :] = np.nanmean(KE_ni_wkb[idx_start:idx_end, :], 0)
        idx_start = idx_end
        month_last = month_now

np.savez('MoorData/ADCP_uv_ni_wkb.npz', u_ni_wkb=u_ni_wkb, v_ni_wkb=v_ni_wkb, KE_ni_wkb=KE_ni_wkb)

plt.figure(1, figsize=(10, 10))
fignums = 0
for i in range(12):
    if i == 5 or i == 6:
        continue
    fignums += 1
    plt.subplot(2, 5, fignums)
    plt.plot(KE_ni_monthly[i, :], depthMoor)
    plt.plot(KE_ni_wkb_monthly[i, :], depthMoor)
    plt.legend(['$KE_{NI}$', '$KE_{NI}^{WKB}$'])
    plt.xlabel('KE $(J/m^{3})$')
    plt.ylabel('depth (m)')
    plt.title('month{}'.format(i+1))
# plt.savefig(r'figures\compare_wkb_monthly.jpg', dpi=300)

plt.figure(2, figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.pcolormesh(dateMoor, depthMoor, KE_ni.T, cmap='Oranges')
plt.clim(0, 10)
plt.colorbar(label='$KE_{NI}$ $(J·m^{-3})$')
plt.ylabel('depth (m)')

plt.subplot(2, 1, 2)
plt.pcolormesh(dateMoor, depthMoor, KE_ni_wkb.T, cmap='Oranges')
plt.clim(0, 10)
plt.colorbar(label='$KE_{NI}^{WKB}$ $(J·m^{-3})$')
plt.ylabel('depth (m)')
# plt.savefig(r'figures/compare_wkb_fullDepth.jpg', dpi=300)
plt.show()
print('cut')
