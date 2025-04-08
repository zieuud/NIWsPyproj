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

fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(4, 2, width_ratios=[40, 1], height_ratios=[1, 1, 1, 1], hspace=0.3, wspace=0.01)

# 第一子图
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(date, tao, 'k-', label='ECMWF')
ax1.legend(loc='upper left')
ax1.set_ylabel(r'$\tau_{NI} (N/m^{2})$')
ax1.set_xticks(date[::30*24])
ax1.xaxis.set_major_formatter(DateFormatter('%b'))

# 第二子图
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax2.plot(date, pi_obs, 'r-', label='Mooring')
ax2.plot(date, pi_slab, 'k-', label='Slab Model')
ax2.legend(loc='upper left')
ax2.set_ylabel(r'$\Pi_{NI}$ $(W/m^{2})$')

# 第三子图
ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
ax3.plot(date, pi_obs_cum, 'r-', label='Mooring')
ax3.plot(date, pi_slab_cum, 'k-', label='Slab Model')
ax3.legend(loc='upper left')
ax3.set_ylabel(r'$\int \Pi_{NI} dt$ $(W/m^{2})$')

# 第四子图
ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
pcm = ax4.pcolormesh(date, depth, ke_ni_wkb.T, cmap='Oranges', vmin=0, vmax=10, shading='auto')
ax4.set_ylabel('Depth (m)')
ax4.set_xlabel('2015 - 2016')

# Colorbar 放在右下角专门区域
cax = fig.add_subplot(gs[3, 1])
cbar = fig.colorbar(pcm, cax=cax)
cbar.set_label(r'$KE_{NI}^{WKB}$ $(J/m^{3})$')

# plt.savefig(r'figures/fig_3_WindWork.jpg', dpi=300, bbox_inches='tight')
plt.show()

