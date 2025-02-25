import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import scipy.stats as stats

vorticity_moor_hourly = np.load(r'ReanaData\GLORYS_vorticity_layers.npy')
strain_moor_hourly = np.load(r'ReanaData\GLORYS_strain_layers.npy')
moorData = np.load(r'ADCP_uv_ni.npz')
KE_ni = moorData['KE_ni']
lat_moor = 36.23
fi = 2 * 7.292e-5 * np.sin(lat_moor/180*np.pi)
vf = vorticity_moor_hourly / fi
# 展平
vorticity_flat = vf.flatten()
KE_flat = KE_ni.flatten()
vorticity_flat = vorticity_flat[~np.isnan(KE_flat)]
KE_flat = KE_flat[~np.isnan(KE_flat)]

plt.figure(figsize=(10, 6))
counts, xedges, yedges, image = plt.hist2d(vorticity_flat, KE_flat, bins=(50, 50), cmap='Blues', density=True,
                                           norm=LogNorm(1e-4, 1))  # 设置对数刻度颜色条
plt.plot([0, 0], [KE_flat.min(), KE_flat.max()], 'k--')
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
plt.xlim(-0.2, 0.2)
# plt.ylim(0, 35)
plt.xlabel(r'$\zeta_g/f$')
plt.ylabel(r'$KE_{NI}^{WKB}$ $(J/m^{3})$')
plt.grid(True)
plt.tight_layout()
# plt.savefig(r'figures\PDF_withDataGLORYS_layers.jpg', dpi=350)
plt.show()
# for debugging
print('c')
