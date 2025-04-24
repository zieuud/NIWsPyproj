import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta


vorticity_moor = np.load(r'ReanaData\AVISO_vorticity1.npy')
adcp = np.load('MoorData/ADCP_uv_ni_wkb.npz')
KE_ni = adcp['KE_ni_wkb']
adcp0 = np.load('MoorData/ADCP_uv.npz')
depth = adcp0['depth_adcp']
moorDate = adcp0['mtime_adcp']
moorDate = [datetime(1, 1, 1) + timedelta(days=m - 367) for m in moorDate]
nz = len(depth)
lat_moor = 36.23
fi = 2 * 7.292e-5 * np.sin(np.deg2rad(lat_moor))
vf = vorticity_moor / fi


fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 0.04], wspace=0.05)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(moorDate, np.zeros(len(moorDate)), 'r--')
ax1.plot(moorDate, vorticity_moor)
ax1.axvspan(moorDate[1], moorDate[1200], color='gray', alpha=0.2)  # event 1
ax1.axvspan(moorDate[1600], moorDate[2650], color='gray', alpha=0.2)  # event 2
ax1.axvspan(moorDate[2800], moorDate[3080], color='gray', alpha=0.2)  # event 3
ax1.axvspan(moorDate[4000], moorDate[4300], color='gray', alpha=0.2)  # event 4
ax1.axvspan(moorDate[4750], moorDate[4900], color='gray', alpha=0.2)  # event 5
ax1.axvspan(moorDate[6080], moorDate[6300], color='gray', alpha=0.2)  # event 6
ax1.set_ylabel(r'$\zeta_g$ $(s^{-1})$')
ax1.set_yticks([-1.5e-5, -1e-5, -0.5e-5, 0, 0.5e-5, 1e-5])
ax1.text(moorDate[20], -1.4e-5, 'a', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

ax2 = fig.add_subplot(gs[1:, 0], sharex=ax1)
c = ax2.pcolormesh(moorDate, depth[:130], KE_ni[:, :130].T, cmap='Oranges', vmin=0, vmax=10, rasterized=True)
ax2.axvspan(moorDate[1], moorDate[1200], color='gray', alpha=0.2)  # event 1
ax2.axvspan(moorDate[1600], moorDate[2650], color='gray', alpha=0.2)  # event 2
ax2.axvspan(moorDate[2800], moorDate[3080], color='gray', alpha=0.2)  # event 3
ax2.axvspan(moorDate[4000], moorDate[4300], color='gray', alpha=0.2)  # event 4
ax2.axvspan(moorDate[4750], moorDate[4900], color='gray', alpha=0.2)  # event 5
ax2.axvspan(moorDate[6080], moorDate[6300], color='gray', alpha=0.2)  # event 6
ax2.set_xlabel('Time')
ax2.set_ylabel('Depth (m)')
ax2.set_ylim([-1000, 0])
ax2.text(moorDate[20], -990, 'b', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

ax3 = fig.add_subplot(gs[1:, 1])
cb = fig.colorbar(c, cax=ax3)
cb.set_label(r'$KE_{NI}^{WKB}$ $(J/m^{3})$')

plt.savefig(r'figuresFinal\fig_9_vorticity_KE.jpg', dpi=300, bbox_inches='tight')
plt.savefig(r'figuresFinal\fig_9_vorticity_KE.png', dpi=300, bbox_inches='tight')
plt.savefig(r'figuresFinal\fig_9_vorticity_KE.pdf', bbox_inches='tight')
plt.show()
