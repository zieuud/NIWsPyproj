import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# ---------- load data ----------
# uvp_mod = np.load(r'MoorData/EnergyFlux_10bcmodes.npz')
# up_mod = uvp_mod['up_mod']
# vp_mod = uvp_mod['vp_mod']
# pp_mod = uvp_mod['pp_mod']
# pp = np.load(r'MoorData/EnergyFlux.npz')['pp']
# pp_mod = pp.reshape(-1, 1, 172)
# fx_mod = pp_mod * up_mod
# fy_mod = pp_mod * vp_mod
# fx_mod = np.load(r'MoorData/EnergyFlux_10bcmodes_fhProj.npz')['fx_mod']
# fy_mod = np.load(r'MoorData/EnergyFlux_10bcmodes_fhProj.npz')['fy_mod']
fx_mod = np.load(r'MoorData/EnergyFlux_10bcmodes_fhProj_1400m.npz')['fx_mod'][:, :, :]
fy_mod = np.load(r'MoorData/EnergyFlux_10bcmodes_fhProj_1400m.npz')['fy_mod'][:, :, :]
# fx_mod = np.load(r'MoorData/EnergyFlux_10bcmodes_fhProj_norm.npz')['fx_mod']
# fy_mod = np.load(r'MoorData/EnergyFlux_10bcmodes_fhProj_norm.npz')['fy_mod']
fx_mod_ta = np.nansum(fx_mod, -1)
fy_mod_ta = np.nansum(fy_mod, -1)
nt, nmodes, nz = np.shape(fx_mod)
dz = 8
dt = 3600


plt.figure(figsize=(12, 12))
for i in range(1, 5):
    plt.subplot(2, 2, i)
    # fx = fx_mod[:, i, :].flatten()
    # fy = fy_mod[:, i, :].flatten()
    fx = fx_mod_ta[:, i].flatten()
    fy = fy_mod_ta[:, i].flatten()
    fx = fx[~np.isnan(fx)]
    fy = fy[~np.isnan(fy)]
    # transfer to angle and magnitude
    theta = np.arctan2(fy, fx)
    magnitude = np.sqrt(fx**2 + fy**2)
    # ---------- plot PDF ----------
    hist, x_edges, y_edges = np.histogram2d(fx, fy, bins=(20, 20))
    hist_density = hist / np.sum(hist)
    x_mesh, y_mesh = np.meshgrid(x_edges, y_edges)
    c = plt.pcolormesh(x_mesh, y_mesh, hist_density.T, cmap='OrRd', norm=LogNorm(vmin=1e-6, vmax=1e-2))
    cb = plt.colorbar(c)
    cb.ax.set_title('PDF')
    plt.plot([-1e3, 1e3], [0, 0], 'k')
    plt.plot([0, 0], [-1e3, 1e3], 'k')
    plt.xlabel('$F_{x}$ ($W/m$)')
    plt.ylabel('$F_{y}$ ($W/m$)')
    # plt.xlim([-2e3, 2e3])
    # plt.ylim([-2e3, 2e3])
    plt.title('mode{}'.format(i))
# plt.savefig(r'figures\fig_8_2_EnergyFluxModesPDF.jpg', dpi=300, bbox_inches='tight')
plt.show()
print('c')
