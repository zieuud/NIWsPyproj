from datetime import datetime, timedelta
import numpy as np
from matplotlib import pyplot as plt

# N2 = np.load(r'ReanaData/WOA23_N2_grid.npy')
N2 = np.load(r'ReanaData/WOA23_stratification_adcpGrid.npz')['Nsq_fusion_adcpGrid']
moorData = np.load('MoorData/ADCP_uv.npz')
depth = moorData['depth_adcp']
moorDate = moorData['mtime_adcp']
dates = [datetime(1, 1, 1) + timedelta(days=m-366) for m in moorDate]

uv_ni = np.load('MoorData/ADCP_uv_ni_ByYu.npz')
u_ni = uv_ni['u_ni']
v_ni = uv_ni['v_ni']
KE_ni = uv_ni['ke_ni']
u_ni_wkb = np.copy(u_ni)
v_ni_wkb = np.copy(v_ni)
N2_averaged = np.nanmean(N2, 1)
max_idx = 180  # 最后一个N2非nan位置
for i in range(len(moorDate)):
    u_ni_wkb[i, :] = u_ni[i, :] * np.sqrt(np.sqrt(N2_averaged[i])/np.sqrt(N2[i, :]))
    v_ni_wkb[i, :] = v_ni[i, :] * np.sqrt(np.sqrt(N2_averaged[i])/np.sqrt(N2[i, :]))
KE_ni_wkb = 1/2*1025*(u_ni_wkb**2+v_ni_wkb**2)

# monthly compose
month_last = 8
month_now = 8
idx_start = 0
idx_end = idx_start
KE_ni_monthly = np.zeros((12, 245))
KE_ni_wkb_monthly = np.zeros((12, 245))
for i in range(len(moorDate)):
    month_now = dates[i].month
    if month_now == month_last and idx_end != 6650-2:
        idx_end = i
    else:
        print(month_last, idx_start, idx_end)
        KE_ni_monthly[month_last-1, :] = np.nanmean(KE_ni[idx_start:idx_end, :], 0)
        KE_ni_wkb_monthly[month_last-1, :] = np.nanmean(KE_ni_wkb[idx_start:idx_end, :], 0)
        idx_start = idx_end
        month_last = month_now

np.savez('MoorData/ADCP_uv_ni_wkb_byFusionNsq.npz', u_ni_wkb=u_ni_wkb, v_ni_wkb=v_ni_wkb, KE_ni_wkb=KE_ni_wkb)
# plt.figure(1, figsize=(4, 5))
# KE_ni_timeAvg = np.nanmean(KE_ni, 0)
# KE_ni_wkb_timeAvg = np.nanmean(KE_ni_wkb, 0)
# plt.plot(KE_ni_timeAvg, depth)
# plt.plot(KE_ni_wkb_timeAvg[:184], depth[:184])
# plt.legend(['$KE_{NI}$', '$KE_{NI}^{WKB}$'])
# plt.xlabel('KE $(J/m^{3})$')
# plt.ylabel('depth (m)')
# plt.tight_layout()
plt.figure(1, figsize=(10, 10))
fignums = 0
for i in range(12):
    if i == 5 or i == 6:
        continue
    fignums += 1
    plt.subplot(2, 5, fignums)
    plt.plot(KE_ni_monthly[i, :], depth)
    plt.plot(KE_ni_wkb_monthly[i, :], depth)
    plt.legend(['$KE_{NI}$', '$KE_{NI}^{WKB}$'])
    plt.xlabel('KE $(J/m^{3})$')
    plt.ylabel('depth (m)')
    plt.title('month{}'.format(i+1))
    # plt.tight_layout()
# plt.savefig(r'figures\compare_wkb_monthly.jpg', dpi=300)
# plt.show()

print('cut')
# plt.figure(1)
# plt.pcolormesh(moorDate, depth, KE_ni.T, cmap='Oranges')
# plt.clim(0, 10)
# plt.colorbar()
#
# plt.figure(2)
# plt.pcolormesh(moorDate, depth, KE_ni_wkb.T, cmap='Oranges')
# plt.colorbar()
# plt.clim(0, 10)
# plt.show()
