import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from scipy import interpolate

vorticity_moor_hourly = np.load(r'ReanaData\AVISO_vorticity3.npy')
strain_moor_hourly = np.load(r'ReanaData\AVISO_strain3.npy')
moorData = np.load(r'ADCP_uv_ni_wkb.npz')
KE_ni = moorData['KE_ni_wkb']
adcp0 = np.load(r'ADCP_uv.npz')
moorDepth = adcp0['depth']
lat_moor = 36.23
fi = 2 * 7.292e-5 * np.sin(lat_moor / 180 * np.pi)
vf = np.tile(vorticity_moor_hourly / fi, (KE_ni.shape[1], 1)).T
sf = np.tile(strain_moor_hourly / fi, (KE_ni.shape[1], 1)).T
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
plt.imshow(hist_density[::-1, :].T, origin='lower', aspect='auto', cmap='Blues',
           extent=[xedges[0], -xedges[0], yedges[0], yedges[-1]], norm=LogNorm(1e-4, 1))
plt.plot(yedges, yedges, 'k--')
plt.plot(-yedges, yedges, 'k--')
# ---------- beautify ----------
plt.colorbar(label='PDF (%)')
plt.xlim(-0.2, 0.2)
plt.ylim(0, 0.2)
plt.xlabel(r'$\zeta/f$')
plt.ylabel(r'$\sigma/f$')
plt.title('jPDF')
# plt.savefig(r'figures\jPDF.jpg', dpi=350)
plt.show()
