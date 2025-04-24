import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, LogNorm
import scipy.stats as stats


vorticity_moor = np.load(r'ReanaData\AVISO_vorticity1.npy')
adcp = np.load('MoorData/ADCP_uv_ni_wkb.npz')
KE_ni = adcp['KE_ni_wkb']
adcp0 = np.load('MoorData/ADCP_uv.npz')
moorDate = adcp0['mtime_adcp']
depth = adcp0['depth_adcp']
nz = len(depth)
lat_moor = 36.23
fi = 2 * 7.292e-5 * np.sin(np.deg2rad(lat_moor))
vf = vorticity_moor / fi

# ---------- plot the PDF ----------
KE_ni_dinteg = np.nansum(KE_ni * 8, 1) / 1000
KE_ni_flat = KE_ni_dinteg.flatten()
# delete nan value
vf_flat = vf[~np.isnan(KE_ni_flat)]
KE_ni_flat = KE_ni_flat[~np.isnan(KE_ni_flat)]
# plot
fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(2, 2, height_ratios=[0.05, 1], width_ratios=[3, 1], hspace=0.1, wspace=0.2)
ax1 = fig.add_subplot(gs[1, 0])
# counts, xedges, yedges, image = ax1.hist2d(vf_flat, KE_ni_flat, bins=(50, 50), cmap='Blues', density=True,
                                           # norm=LogNorm(vmin=1e-2, vmax=1))
hist, xedges, yedges = np.histogram2d(vf_flat, KE_ni_flat, bins=[50, 50])
hist_density = hist/np.max(hist)
pcm = ax1.pcolormesh(xedges, yedges, hist_density.T, cmap='Blues', norm=LogNorm(vmin=1e-2, vmax=1))
ax1.plot([0, 0], [KE_ni_flat.min(), KE_ni_flat.max()], 'k--')
# ---------- find the mean values of every bins ----------
maxIdx = np.zeros(50)
for i in range(50):
    if np.sum(hist[:, i]) == 0:
        maxIdx[i] = np.argmin(abs(xedges[1:]))
    else:
        maxIdx[i] = np.argmin(abs(xedges[1:] - np.average(xedges[1:], weights=hist[:, i] / np.sum(hist[:, i]))))
maxIdx = maxIdx.astype(int)
ax1.plot(xedges[maxIdx], yedges[1:], 'r-')
# ---------- fit the mean values of every bins ----------
slope, intercept, r_value, p_value, std_err = stats.linregress(yedges[1:], xedges[maxIdx])
fitted = slope * yedges[1:] + intercept
ax1.plot(fitted, yedges[1:], 'r--')
ax1.text(xedges[1], yedges[-5], 'r={:.2f}'.format(r_value))
ax1.text(-0.2, 0, 'a', ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
# ---------- beautify ----------
ax1.set_xlim(-0.2, 0.2)
ax1.set_xlabel(r'$\zeta_g/f$')
ax1.set_ylabel(r'$KE_{NI}^{WKB}$ $(kJ/m^{3})$')
ax1.grid(True)
# plot the colorbar
cax = fig.add_subplot(gs[0, 0])
cb = fig.colorbar(pcm, cax=cax, orientation='horizontal')
cb.ax.xaxis.set_label_position('top')
cb.ax.xaxis.tick_top()
cb.set_label('PDF (%)', loc='center')

# plot the vertical profile
ax2 = fig.add_subplot(gs[1, 1])
# --------- calculate KE_ni_wkb with negative and positive vorticity ----------
positiveVor = np.zeros(len(depth))
negativeVor = np.zeros(len(depth))
for idx in range(len(vorticity_moor)):
    if (vorticity_moor[idx] > 0) & (~np.isnan(KE_ni[idx, :]).all()):
        validIndices = ~np.isnan(KE_ni[idx, :])
        positiveVor[validIndices] += KE_ni[idx, validIndices]
    elif (vorticity_moor[idx]) < 0 & (~np.isnan(KE_ni[idx, :]).all()):
        validIndices = ~np.isnan(KE_ni[idx, :])
        negativeVor[validIndices] += KE_ni[idx, validIndices]
    elif np.isnan(KE_ni[idx, :]).all():
        print('nan value of KE_ni!')
    else:
        print('0 vorticity!')
positiveVor[positiveVor == 0] = np.nan
negativeVor[negativeVor == 0] = np.nan
timeSpan = moorDate[-1] - moorDate[0]
ax2.plot(positiveVor / timeSpan, depth/1000, 'b-')
ax2.plot(negativeVor / timeSpan, depth/1000, 'r-')
ax2.legend([r'$\zeta_{g} > 0$', r'$\zeta_{g} < 0$'])
ax2.set_ylabel('Depth (km)', labelpad=-2)
ax2.set_xlabel('Time-averaged $KE_{ni}^{wkb}$ ($J/m^{3}$)')
ax2.set_ylim([-2, 0])
ax2.set_yticks([-2, -1.5, -1, -0.5, 0])
ax2.text(0, -2.0, 'b', ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

plt.savefig(r'figures/fig_9_PDF.jpg', dpi=300, bbox_inches='tight')
plt.savefig(r'figuresFinal/fig_9_PDF.png', dpi=300, bbox_inches='tight')
plt.savefig(r'figuresFinal/fig_9_PDF.pdf', bbox_inches='tight')
plt.show()
