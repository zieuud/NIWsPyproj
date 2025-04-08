import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# ---------- load data ----------
# uvp_ni_wkb = np.load(r'ReanaData\WOA23_uvp_ni_wkb.npz')
# up_ni_wkb = uvp_ni_wkb['up_ni_wkb']
# vp_ni_wkb = uvp_ni_wkb['vp_ni_wkb']
# pp = np.load(r'ReanaData\WOA23_pp.npy')
# fx = pp * up_ni_wkb
# fy = pp * vp_ni_wkb
flux = np.load(r'MoorData/EnergyFlux.npz')
fx = flux['fx']
fy = flux['fy']
nt, nz = np.shape(fx)
dz = 8
dt = 3600

# fx = fx.flatten()
# fy = fy.flatten()
fx = np.nansum(fx * dz, 1)
fy = np.nansum(fy * dz, 1)
fx = fx[~np.isnan(fx)]
fy = fy[~np.isnan(fy)]
# transfer to angle and magnitude
theta = np.arctan2(fy, fx)
magnitude = np.sqrt(fx**2 + fy**2)
# ---------- plot PDF ----------
hist, theta_edges, r_edges = np.histogram2d(theta, magnitude, bins=(36, 20))
hist_density = hist / np.sum(hist)
theta_mesh, magnitude_mesh = np.meshgrid(theta_edges, r_edges)
plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
c = ax.pcolormesh(theta_mesh, magnitude_mesh, hist_density.T, cmap='Reds', norm=LogNorm(vmin=1e-4, vmax=1e-1))
ax.set_yticks([500, 1000, 1500])
plt.colorbar(c, label='PDF')
plt.title('PDF of Horizontal Energy Flux')
# plt.savefig(r'figures\PDF of Horizontal Energy Flux2.jpg', dpi=300)
plt.show()
print('c')
