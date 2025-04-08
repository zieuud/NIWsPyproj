import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# ---------- load data ----------
fh = np.load(r'MoorData\EnergyFlux.npz')
fx = fh['fx'].flatten()
fy = fh['fy'].flatten()
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
c = ax.pcolormesh(theta_mesh, magnitude_mesh, hist_density.T, cmap='Reds', norm=LogNorm(vmin=1e-6, vmax=1e-2))
ax.set_yticks([50, 100, 150])
plt.colorbar(c, label='PDF')
plt.title('PDF of Horizontal Energy Flux')
# plt.savefig(r'figures\PDF of Horizontal Energy Flux.jpg', dpi=300)
plt.show()
print('c')
