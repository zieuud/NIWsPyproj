import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter


data1 = np.load(r'ReanaData/ERA5_tao_ni.npz')
tao_x_ni = data1['tao_x_ni']
tao_y_ni = data1['tao_y_ni']
tao = np.sqrt(tao_x_ni ** 2 + tao_y_ni ** 2)
data2 = np.load(r'ReanaData/ERA5_windWork.npz')
pi_obs = data2['windWork_ni_obs']
pi_slab = data2['windWork_ni_slab']
data3 = np.load(r'ReanaData/ERA5_windWork_cum.npz')
pi_obs_cum = data3['windWork_ni_obs_cum']
pi_slab_cum = data3['windWork_ni_slab_cum']
data4 = np.load(r'MoorData/ADCP_uv_ni_wkb.npz')
ke_ni_wkb = data4['KE_ni_wkb']
data5 = np.load(r'MoorData/ADCP_uv.npz')
time = data5['mtime_adcp']
depth = data5['depth_adcp']
date = [datetime(1, 1, 1) + timedelta(days=i - 367) for i in time]
data6 = np.load(r'ReanaData/GLORYS_pycnocline.npy')
pycnoline = -data6

fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(4, 2, width_ratios=[40, 1], height_ratios=[1, 1, 1, 1], hspace=0.3, wspace=0.01)

# 第一子图
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(date, tao, 'k-', label='ECMWF')
ax1.axvspan(date[1], date[1200], color='gray', alpha=0.2)  # event 1
ax1.axvspan(date[1600], date[2650], color='gray', alpha=0.2)  # event 2
ax1.axvspan(date[2800], date[3080], color='gray', alpha=0.2)  # event 3
ax1.axvspan(date[4000], date[4300], color='gray', alpha=0.2)  # event 4
ax1.axvspan(date[4750], date[4900], color='gray', alpha=0.2)  # event 5
ax1.axvspan(date[6080], date[6300], color='gray', alpha=0.2)  # event 6
ax1.legend(loc='upper left')
ax1.set_ylabel(r'$\tau_{NI} (N/m^{2})$')
ax1.set_xticks(date[::30*24])
ax1.xaxis.set_major_formatter(DateFormatter('%b'))
ax1.text(date[int(6650 - 0.98 * 6650)], 0, 'a', ha='left', va='bottom', fontsize=12,
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))


# 第二子图
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax2.plot(date, pi_obs, 'r-', label='Mooring')
ax2.plot(date, pi_slab, 'k-', label='Slab Model')
ax2.axvspan(date[1], date[1200], color='gray', alpha=0.2)  # event 1
ax2.axvspan(date[1600], date[2650], color='gray', alpha=0.2)  # event 2
ax2.axvspan(date[2800], date[3080], color='gray', alpha=0.2)  # event 3
ax2.axvspan(date[4000], date[4300], color='gray', alpha=0.2)  # event 4
ax2.axvspan(date[4750], date[4900], color='gray', alpha=0.2)  # event 5
ax2.axvspan(date[6080], date[6300], color='gray', alpha=0.2)  # event 6
ax2.legend(loc='upper left')
ax2.set_ylabel(r'$\Pi_{NI}$ $(W/m^{2})$')
ax2.text(date[int(6650 - 0.98 * 6650)], -0.008, 'b', ha='left', va='bottom', fontsize=12,
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

# 第三子图
ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
ax3.plot(date, pi_obs_cum, 'r-', label='Mooring')
ax3.plot(date, pi_slab_cum, 'k-', label='Slab Model')
ax3.axvspan(date[1], date[1200], color='gray', alpha=0.2)  # event 1
ax3.axvspan(date[1600], date[2650], color='gray', alpha=0.2)  # event 2
ax3.axvspan(date[2800], date[3080], color='gray', alpha=0.2)  # event 3
ax3.axvspan(date[4000], date[4300], color='gray', alpha=0.2)  # event 4
ax3.axvspan(date[4750], date[4900], color='gray', alpha=0.2)  # event 5
ax3.axvspan(date[6080], date[6300], color='gray', alpha=0.2)  # event 6
ax3.legend(loc='upper left')
ax3.set_ylabel(r'$\int \Pi_{NI} dt$ $(W/m^{2})$')
ax3.text(date[int(6650 - 0.98 * 6650)], 0, 'c', ha='left', va='bottom', fontsize=12,
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
ax3.text(date[600], 6, '1', ha='center', va='center')
ax3.text(date[2125], 6, '2', ha='center', va='center')
ax3.text(date[2940], 6, '3', ha='center', va='center')
ax3.text(date[4150], 6, '4', ha='center', va='center')
ax3.text(date[4825], 6, '5', ha='center', va='center')
ax3.text(date[6190], 6, '6', ha='center', va='center')

# 第四子图
ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
pcm = ax4.pcolormesh(date, depth, ke_ni_wkb.T, cmap='Oranges', vmin=0, vmax=10, shading='auto', rasterized=True)
ax4.plot(date, pycnoline, 'b-', label='$M_{ML}$', linewidth=1)
ax4.axvspan(date[1], date[1200], color='gray', alpha=0.2)  # event 1
ax4.axvspan(date[1600], date[2650], color='gray', alpha=0.2)  # event 2
ax4.axvspan(date[2800], date[3080], color='gray', alpha=0.2)  # event 3
ax4.axvspan(date[4000], date[4300], color='gray', alpha=0.2)  # event 4
ax4.axvspan(date[4750], date[4900], color='gray', alpha=0.2)  # event 5
ax4.axvspan(date[6080], date[6300], color='gray', alpha=0.2)  # event 6
ax4.set_ylabel('Depth (m)')
ax4.set_xlabel('2015 - 2016')
ax4.legend(loc='lower right')
ax4.set_ylim([-2000, 0])
ax4.text(date[int(6650 - 0.98 * 6650)], -1900, 'd', ha='left', va='bottom', fontsize=12,
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

# Colorbar 放在右下角专门区域
cax = fig.add_subplot(gs[3, 1])
cbar = fig.colorbar(pcm, cax=cax)
cbar.set_label(r'$KE_{NI}$ $(J/m^{3})$')

plt.savefig(r'figures/fig_4_WindWork.jpg', dpi=300, bbox_inches='tight')
plt.savefig(r'figuresFinal/fig_4_WindWork.png', dpi=300, bbox_inches='tight')
plt.savefig(r'figuresFinal/fig_4_WindWork.pdf', bbox_inches='tight')
plt.show()
