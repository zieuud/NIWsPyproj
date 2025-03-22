import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# ---------- load data ----------
uvp_ni_wkb_modes = np.load(r'ReanaData\WOA23_uvp_ni_wkb_10modes.npz')
up_ni_wkb_modes = uvp_ni_wkb_modes['up_ni_wkb_mod']
vp_ni_wkb_modes = uvp_ni_wkb_modes['vp_ni_wkb_mod']
pp = np.load(r'ReanaData\WOA23_pp.npy')
nt, nmodes, nz = np.shape(up_ni_wkb_modes)
dz = 8
dt = 3600

fx_modes = np.copy(up_ni_wkb_modes) * np.nan
fy_modes = np.copy(up_ni_wkb_modes) * np.nan
for i in range(nmodes):
    fx_modes[:, i, :] = pp * np.squeeze(up_ni_wkb_modes[:, i, :])
    fy_modes[:, i, :] = pp * np.squeeze(vp_ni_wkb_modes[:, i, :])

plt.figure(figsize=(12, 12))
for i in range(1, 5):
    plt.subplot(2, 2, i)
    print(i)
    fx = np.nansum(fx_modes[:, i, :], 1) * dz
    fy = np.nansum(fy_modes[:, i, :], 1) * dz
    fx = fx[~np.isnan(fx)]
    fy = fy[~np.isnan(fy)]
    # transfer to angle and magnitude
    theta = np.arctan2(fy, fx)
    magnitude = np.sqrt(fx**2 + fy**2)
    # ---------- plot PDF ----------
    hist, x_edges, y_edges = np.histogram2d(fx, fy, bins=(36, 36))
    hist_density = hist / np.sum(hist)
    x_mesh, y_mesh = np.meshgrid(x_edges, y_edges)
    c = plt.pcolormesh(x_mesh, y_mesh, hist_density.T, cmap='Reds', norm=LogNorm(vmin=1e-4, vmax=1))
    cb = plt.colorbar(c)
    cb.ax.set_title('PDF')
    plt.plot([-1e3, 1e3], [0, 0], 'k')
    plt.plot([0, 0], [-1e3, 1e3], 'k')
    plt.xlabel('$F_{x}$ ($W·m^{-1}$)')
    plt.ylabel('$F_{y}$ ($W·m^{-1}$)')
    plt.xlim([-1e3, 1e3])
    plt.ylim([-1e3, 1e3])
    plt.title('mode{}'.format(i))
plt.savefig(r'figures\PDF of Modes Horizontal Energy Flux.jpg', dpi=300)
plt.show()
print('c')
