from datetime import datetime, timedelta

import numpy as np
from matplotlib import pyplot as plt

N2 = np.load(r'ReanaData\WOA23_N2_grid.npy')
moorData = np.load('ADCP_uv.npz')
depth = moorData['depth']
moorDate = moorData['mtime']
dates = [datetime(1, 1, 1) + timedelta(days=m-366) for m in moorDate]

uv_ni = np.load('ADCP_uv_ni.npz')
u_ni = uv_ni['u_ni']
v_ni = uv_ni['v_ni']
KE_ni = uv_ni['KE_ni']
u_ni_wkb = np.copy(u_ni)
v_ni_wkb = np.copy(v_ni)
N2_averaged = np.nanmean(N2, 1)

for i in range(len(moorDate)):
    ml_idx = np.nanargmax(N2[i, :])
    u_ni_wkb[i, ml_idx:184] = u_ni[i, ml_idx:184] * np.sqrt(np.sqrt(N2_averaged[i])/np.sqrt(N2[i, ml_idx:184]))
    v_ni_wkb[i, ml_idx:184] = v_ni[i, ml_idx:184] * np.sqrt(np.sqrt(N2_averaged[i])/np.sqrt(N2[i, ml_idx:184]))
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
# np.savez('ADCP_uv_ni_wkb.npz', u_ni_wkb=u_ni_wkb, v_ni_wkb=v_ni_wkb, KE_ni_wkb=KE_ni_wkb)
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
    plt.plot(KE_ni_wkb_monthly[i, :184], depth[:184])
    plt.legend(['$KE_{NI}$', '$KE_{NI}^{WKB}$'])
    plt.xlabel('KE $(J/m^{3})$')
    plt.ylabel('depth (m)')
    plt.title('month{}'.format(i+1))
    # plt.tight_layout()
plt.savefig(r'figures\compare_wkb_monthly.jpg', dpi=300)
plt.show()

print('cut')
# [depth_axis1, time_axis1] = np.meshgrid(depth, moorDate)
# plt.figure(1)
# plt.pcolor(time_axis1, depth_axis1, KE_ni, cmap='Oranges')
# plt.clim(0, 10)
# plt.colorbar()

# [depth_axis2, time_axis2] = np.meshgrid(depth, moorDate)
# plt.figure(2)
# plt.pcolor(time_axis2, depth_axis2, KE_ni_wkb, cmap='Oranges')
# plt.colorbar()
# plt.clim(0, 10)
# plt.show()
