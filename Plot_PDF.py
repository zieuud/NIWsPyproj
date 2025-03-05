import matplotlib.pyplot as plt
import numpy as np
import gsw
from matplotlib.colors import Normalize, LogNorm
import scipy.stats as stats

# vorticity_moor = np.load(r'ReanaData\Glorys_vorticity.npy')
# N2 = np.load(r'ReanaData\WOA23_N2_grid.npy')
temp = np.load(r'ReanaData\WOA23_temp_grid.npy')
vorticity_moor = np.load(r'ReanaData\AVISO_vorticity4.npy')
adcp = np.load('ADCP_uv_ni_wkb.npz')
KE_ni = adcp['KE_ni_wkb']
adcp0 = np.load('ADCP_uv.npz')
moorDate = adcp0['mtime']
depth = adcp0['depth']
lat_moor = 36.23
fi = 2 * 7.292e-5 * np.sin(lat_moor/180*np.pi)
vf = vorticity_moor / fi
# ----------plot the PDF----------
KE_ni_dinteg = np.zeros(np.size(KE_ni, 0))
for idx in range(np.size(KE_ni, 0)):
    # ml_idx = np.nanargmax(N2[idx, :])
    ml_idx = np.argwhere(temp[idx, :] - temp[idx, 0] < -0.5)[0][0]
    KE_ni_under_ml = KE_ni[idx, ml_idx:]
    depth_under_ml = -depth[ml_idx:]
    KE_ni_dinteg[idx] = np.trapz(KE_ni_under_ml[~np.isnan(KE_ni_under_ml)], depth_under_ml[~np.isnan(KE_ni_under_ml)]) / 1000
KE_ni_flat = KE_ni_dinteg.flatten()
# KE_ni_flat = KE_ni.flatten()
# vf = np.tile(vf, KE_ni.shape[1])
vf_flat = vf[~np.isnan(KE_ni_flat)]
KE_ni_flat = KE_ni_flat[~np.isnan(KE_ni_flat)]

plt.figure(figsize=(10, 6))
counts, xedges, yedges, image = plt.hist2d(
    vf_flat,
    KE_ni_flat,
    bins=(50, 50),
    cmap='Blues',
    density=True,
    norm=LogNorm(vmin=1e-2, vmax=1)  # 设置对数刻度颜色条
)
plt.plot([0, 0], [KE_ni_flat.min(), KE_ni_flat.max()], 'k--')
# plt.show()
density = np.transpose(counts/np.sum(counts))
density = density/np.max(density)
plt.pcolor(xedges, yedges, density, cmap='Blues', norm=LogNorm(vmin=1e-2, vmax=1))
# plt.colorbar(norm=LogNorm(vmin=1e-2, vmax=1))
# plt.show()
# ---------- find the mean values of every bins ----------
maxIdx = np.zeros(50)
for i in range(50):
    if np.sum(counts[:, i]) == 0:
        maxIdx[i] = np.argmin(abs(xedges[1:]))
    else:
        maxIdx[i] = np.argmin(abs(xedges[1:] - np.average(xedges[1:], weights=counts[:, i] / np.sum(counts[:, i]))))
maxIdx = maxIdx.astype(int)
plt.plot(xedges[maxIdx], yedges[1:], 'r-')
# ---------- fit the mean values of every bins ----------
slope, intercept, r_value, p_value, std_err = stats.linregress(yedges[1:], xedges[maxIdx])
fitted = slope * yedges[1:] + intercept
plt.plot(fitted, yedges[1:], 'r--')
plt.text(xedges[1], yedges[-5], 'r={:.2f}'.format(r_value))
# ---------- beautify ----------
cb = plt.colorbar()
cb.set_label('PDF (%)')
plt.xlim(-0.1, 0.1)
# plt.ylim(0, 35)
plt.xlabel(r'$\zeta_g/f$')
plt.ylabel(r'$KE_{NI}^{WKB}$ $(J/m^{3})$')
plt.grid(True)
plt.tight_layout()
plt.savefig(r'figures\PDF_depth_integrated4.jpg', dpi=300)
plt.show()
# # ----------plot the vorticity and KE_ni---------
# fig = plt.figure(figsize=(10, 8))
# gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 0.05], wspace=0.05)
#
# ax1 = fig.add_subplot(gs[0, 0])
# ax1.plot(moorDate, np.zeros(len(moorDate)), 'r--')
# ax1.plot(moorDate, vorticity_moor)
# ax1.set_ylabel(r'$\zeta_g$ $(s^{-1})$')
#
# ax2 = fig.add_subplot(gs[1:, 0], sharex=ax1)
# [depth_mesh, moorDate_mesh] = np.meshgrid(depth[:120], moorDate)
# c = ax2.pcolor(moorDate_mesh, depth_mesh, KE_ni[:, :120], cmap='Oranges', vmin=0, vmax=10)
# ax2.set_xlabel('time')
# ax2.set_ylabel('depth (m)')
#
# ax3 = fig.add_subplot(gs[1:, 1])
# cb = fig.colorbar(c, cax=ax3)
# cb.set_label(r'$KE_{NI}^{WKB}$ $(J/m^{3})$')
#
# plt.tight_layout()
# plt.savefig(r'figures\vorticity_KE.jpg', dpi=300)
# plt.show()

print('c')
