import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from scipy import interpolate

vorticity_moor_hourly = np.load(r'ReanaData\AVISO_vorticity1.npy')
strain_moor_hourly = np.load(r'ReanaData\AVISO_strain1.npy')
moorData = np.load(r'ADCP_uv_ni_wkb.npz')
KE_ni = moorData['KE_ni_wkb']
adcp0 = np.load(r'ADCP_uv.npz')
moorDepth = adcp0['depth']
lat_moor = 36.23
fi = 2 * 7.292e-5 * np.sin(lat_moor / 180 * np.pi)
vf = np.tile(vorticity_moor_hourly / fi, (KE_ni.shape[1], 1)).T
sf = np.tile(strain_moor_hourly / fi, (KE_ni.shape[1], 1)).T
# vf = vorticity_moor_hourly / fi
# sf = strain_moor_hourly / fi
# KE_ni = np.nansum(KE_ni, 1)
KE_flat = KE_ni.flatten()
vorticity_flat = vf.flatten()[~np.isnan(KE_flat)]
strain_flat = sf.flatten()[~np.isnan(KE_flat)]
KE_flat = KE_flat[~np.isnan(KE_flat)]

x_bins = np.linspace(vorticity_flat.min(), vorticity_flat.max(), 50)  # vorticity的区间
y_bins = np.linspace(strain_flat.min(), strain_flat.max(), 50)  # strain的区间

hist, xedges, yedges = np.histogram2d(vorticity_flat, strain_flat, bins=[x_bins, y_bins], weights=KE_flat)
hist_density = hist / (np.nanmax(hist) - np.nanmin(hist))
# 绘制热图
plt.figure(figsize=(12, 6))
plt.imshow(hist_density.T, origin='lower', aspect='auto', cmap='Blues',
           extent=(xedges[0], -xedges[0], yedges[0], yedges[-1]), norm=LogNorm(1e-4, 1))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0, -1], [0, 1], 'k--')
# ---------- beautify ----------
plt.colorbar(label='PDF (%)')
plt.xlim(-0.1, 0.1)
plt.ylim(0, 0.1)
plt.xlabel(r'$\zeta/f$')
plt.ylabel(r'$\sigma/f$')
plt.title('jPDF')
# plt.savefig(r'figures\jPDF_withAVISO1.jpg', dpi=350)
plt.show()
