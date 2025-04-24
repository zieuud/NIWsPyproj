import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA
from scipy import stats


def cal_confidence(data):
    n = len(data)
    std = np.nanstd(data, ddof=1)
    se = std / np.sqrt(n)
    confidence = 0.99
    alpha = 1 - confidence
    t_value = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margin = t_value * se
    return margin


# ---------- load data ----------
fx_mod = np.load(r'MoorData/EnergyFlux_modes.npz')['fx_mod'][:3960, :, :]
fy_mod = np.load(r'MoorData/EnergyFlux_modes.npz')['fy_mod'][:3960, :, :]
fx_mod = np.nansum(fx_mod, -1)
fy_mod = np.nansum(fy_mod, -1)
nt, nmodes = np.shape(fx_mod)
dz = 8
dt = 3600


plt.figure(figsize=(12, 12))
for i in range(1, 5):
    plt.subplot(2, 2, i)
    fx = fx_mod[:, i].flatten()
    fy = fy_mod[:, i].flatten()
    fx = fx[~np.isnan(fx)]
    fy = fy[~np.isnan(fy)]
    fh = np.sqrt(fx ** 2 + fy ** 2)
    # PCA analysis for obtain the main direction of energy flux
    data = np.vstack((fx, fy)).T
    pca = PCA(n_components=2)
    pca.fit(data)
    center = np.mean(data, axis=0)
    direction = pca.components_[0]
    direction = -direction if np.mean(data @ direction) < 0 else direction
    amplitude = np.mean(fh) * 10
    # calculate the direction error
    width = 1.96 * np.sqrt(pca.explained_variance_[0] / len(fx))
    spread_angle = np.degrees(np.arctan2(width, 1))
    theta_arrow = np.arctan2(direction[1], direction[0])
    theta_left = theta_arrow - np.radians(spread_angle)
    theta_right = theta_arrow + np.radians(spread_angle)
    x_left = center[0] + amplitude * np.cos(theta_left)
    y_left = center[1] + amplitude * np.sin(theta_left)
    x_right = center[0] + amplitude * np.cos(theta_right)
    y_right = center[1] + amplitude * np.sin(theta_right)
    # plot PDF
    hist, x_edges, y_edges = np.histogram2d(fx, fy, bins=(20, 20))
    # hist_density = hist / np.sum(hist)
    hist_density = (hist - np.nanmin(hist)) / (np.nanmax(hist) - np.nanmin(hist))
    c = plt.pcolormesh(x_edges, y_edges, hist_density.T, cmap='bone_r', norm=LogNorm(vmin=5e-4, vmax=1))
    cb = plt.colorbar(c)
    cb.ax.set_title('PDF')
    if i == 1:
        # plot the main axis
        ax = plt.gca()
        ax.annotate('', xy=(amplitude * direction[0], amplitude * direction[1]), xytext=(center[0], center[1]),
                    arrowprops=dict(facecolor='#0074e4', edgecolor='#0074e4', width=2, headwidth=5),
                    annotation_clip=False, zorder=3)
        # plot the direction error
        plt.plot([center[0], x_left], [center[1], y_left], color='#0074e4', linestyle='--', zorder=3)
        plt.plot([center[0], x_right], [center[1], y_right], color='#0074e4', linestyle='--', zorder=3)
        # plot the grid
        plt.plot([-3e2, 3e2], [0, 0], 'k', zorder=2)
        plt.plot([0, 0], [-3e2, 3e2], 'k', zorder=2)
        plt.xlim([-3e2, 3e2])
        plt.ylim([-3e2, 3e2])
    else:
        # plot the main axis
        ax = plt.gca()
        ax.annotate('', xy=(amplitude * direction[0], amplitude * direction[1]), xytext=(center[0], center[1]),
                    arrowprops=dict(facecolor='#0074e4', edgecolor='#0074e4', width=1, headwidth=3),
                    annotation_clip=False, zorder=3)
        # plot the direction error
        plt.plot([center[0], x_left], [center[1], y_left], color='#0074e4', linestyle='--', zorder=3)
        plt.plot([center[0], x_right], [center[1], y_right], color='#0074e4', linestyle='--', zorder=3)
        # plot the grid
        plt.plot([-1e2, 1e2], [0, 0], 'k', zorder=2)
        plt.plot([0, 0], [-1e2, 1e2], 'k', zorder=2)
        plt.xlim([-1e2, 1e2])
        plt.ylim([-1e2, 1e2])
    plt.xlabel('$F_{x}$ ($W/m$)')
    plt.ylabel('$F_{y}$ ($W/m$)')
    plt.title('mode{}'.format(i))

plt.savefig(r'figures\fig_8_2_EnergyFluxModesPDF.jpg', dpi=300, bbox_inches='tight')
plt.show()
print('c')
