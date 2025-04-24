import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib.gridspec as gridspec


def cal_confidence(data):
    n = len(data)
    std = np.nanstd(data, ddof=1)
    se = std / np.sqrt(n)
    confidence = 0.99
    alpha = 1 - confidence
    t_value = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margin = t_value * se
    return margin


fig = plt.figure(figsize=(20, 10))
plt.rcParams['font.size'] = 16
gs = gridspec.GridSpec(2, 4, width_ratios=[2, 1, 1, 0.1], height_ratios=[1, 1], hspace=0.3, wspace=0.3)
figIdx = ['a', 'b', 'c', 'd', 'e']
# ---------------------------------------- total energy flux ----------------------------------------
# ---------- load data ----------
fh_mod = np.load(r'MoorData/EnergyFlux_modes.npz')
fx = np.nansum(fh_mod['fx_mod'][:3960, :, :], (1, 2))  # mode fit
fy = np.nansum(fh_mod['fy_mod'][:3960, :, :], (1, 2))  # cut before date[3960]
fx = fx[~np.isnan(fx)]
fy = fy[~np.isnan(fy)]
fh = np.sqrt(fx ** 2 + fy ** 2)
# transfer to angle and magnitude for plotting on polar coordinate
theta = np.arctan2(fy, fx)
magnitude = np.sqrt(fx ** 2 + fy ** 2)
# PCA analysis for obtain the main direction of energy flux
data = np.vstack((fx, fy)).T
pca = PCA(n_components=2)
pca.fit(data)
direction = pca.components_[0]
direction = -direction if np.mean(data @ direction) < 0 else direction
theta_arrow = np.arctan2(direction[1], direction[0])
r_arrow = np.mean(fh) * 10  # enlarge 10 times for clarity
# calculate the theta error
varianceRatio = pca.explained_variance_ratio_[0]
width = 1.96 * np.sqrt(pca.explained_variance_[0] / len(fx))  # 1.96 to estimate the 95% confidence interval
spread_angle = np.arctan2(width, 1)
theta_left = theta_arrow - spread_angle
theta_right = theta_arrow + spread_angle
hist, theta_edges, r_edges = np.histogram2d(theta, magnitude, bins=(36, 20))
hist_density = (hist - np.nanmin(hist)) / (np.nanmax(hist) - np.nanmin(hist))
theta_mesh, magnitude_mesh = np.meshgrid(theta_edges, r_edges)
ax1 = fig.add_subplot(gs[:, 0], polar=True)
c = ax1.pcolormesh(theta_mesh, magnitude_mesh, hist_density.T, cmap='OrRd', norm=LogNorm(vmin=5e-3, vmax=1))
ax1.set_yticks([200, 400, 600])
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
cbar = fig.colorbar(c, label='PDF', ax=ax1, orientation='horizontal', pad=0.01)
cbar.ax.tick_params(labelsize=14)
# add the arrow of main axis
ax1.annotate('', xy=(theta_arrow, r_arrow), xytext=(0, 0),
             arrowprops=dict(facecolor='#0074e4', edgecolor='#0074e4', width=2, headwidth=5),
             annotation_clip=False, zorder=2)
# add the error on theta
ax1.plot([theta_left, theta_left],
         [0, r_arrow], linestyle='--', color='#0074e4', linewidth=1.5)
ax1.plot([theta_right, theta_right],
         [0, r_arrow], linestyle='--', color='#0074e4', linewidth=1.5)
ax1.set_title('PDF of Total Horizontal Energy Flux')
ax1.text(np.pi / 2, 600, 'a', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
ax1.text(np.pi / 2, 500, '{:.2f}% variance explained'.format(varianceRatio * 100),
         ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
# ---------------------------------------- modes energy flux ----------------------------------------
# ---------- load data ----------
fx_mod = np.load(r'MoorData/EnergyFlux_modes.npz')['fx_mod'][:3960, :, :]
fy_mod = np.load(r'MoorData/EnergyFlux_modes.npz')['fy_mod'][:3960, :, :]
fx_mod = np.nansum(fx_mod, -1)
fy_mod = np.nansum(fy_mod, -1)
nt, nmodes = np.shape(fx_mod)
dz = 8
dt = 3600
loc = [[0, 1], [0, 2], [1, 1], [1, 2]]
for i in range(1, 5):
    ax = fig.add_subplot(gs[loc[i - 1][0], loc[i - 1][1]])
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
    varianceRatio = pca.explained_variance_ratio_[0]
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
    c = ax.pcolormesh(x_edges, y_edges, hist_density.T, cmap='bone_r', norm=LogNorm(vmin=5e-4, vmax=1))
    # fig.colorbar(c, ax=ax)
    # cb.ax.set_title('PDF')
    if i == 1:
        # plot the main axis
        ax.annotate('', xy=(amplitude * direction[0], amplitude * direction[1]), xytext=(center[0], center[1]),
                    arrowprops=dict(facecolor='#f06868', edgecolor='#f06868', width=2, headwidth=5),
                    annotation_clip=False, zorder=3)
        # plot the direction error
        ax.plot([center[0], x_left], [center[1], y_left], color='#f06868', linestyle='--', zorder=3)
        ax.plot([center[0], x_right], [center[1], y_right], color='#f06868', linestyle='--', zorder=3)
        # plot the grid
        ax.plot([-3e2, 3e2], [0, 0], 'k', zorder=2)
        ax.plot([0, 0], [-3e2, 3e2], 'k', zorder=2)
        ax.set_xlim([-3e2, 3e2])
        ax.set_ylim([-3e2, 3e2])
        ax.set_xticks([-300, -150, 0, 150, 300])
        ax.set_yticks([-300, -150, 0, 150, 300])
        ax.text(-300 + 30, 300 - 30, figIdx[i], ha='left', va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
        ax.text(-300 + 30, 300 - 90, '{:.2f}% variance explained'.format(varianceRatio * 100),
                ha='left', va='top', fontsize=12, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    else:
        # plot the main axis
        ax.annotate('', xy=(amplitude * direction[0], amplitude * direction[1]), xytext=(center[0], center[1]),
                    arrowprops=dict(facecolor='#f06868', edgecolor='#f06868', width=1, headwidth=3),
                    annotation_clip=False, zorder=3)
        # plot the direction error
        ax.plot([center[0], x_left], [center[1], y_left], color='#f06868', linestyle='--', zorder=3)
        ax.plot([center[0], x_right], [center[1], y_right], color='#f06868', linestyle='--', zorder=3)
        # plot the grid
        ax.plot([-1e2, 1e2], [0, 0], 'k', zorder=2)
        ax.plot([0, 0], [-1e2, 1e2], 'k', zorder=2)
        ax.set_xlim([-1e2, 1e2])
        ax.set_ylim([-1e2, 1e2])
        ax.set_xticks([-100, -50, 0, 50, 100])
        ax.set_yticks([-100, -50, 0, 50, 100])
        ax.text(-100 + 10, 100 - 10, figIdx[i], ha='left', va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
        ax.text(-100 + 10, 100 - 30, '{:.2f}% variance explained'.format(varianceRatio * 100),
                ha='left', va='top', fontsize=12, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    if i == 1 or i == 3:
        ax.set_ylabel('$F_{y}$ ($W/m$)', labelpad=-10)
    if i == 3 or i == 4:
        ax.set_xlabel('$F_{x}$ ($W/m$)', labelpad=0)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_title('mode {}'.format(i))

cax = fig.add_subplot(gs[:, 3])
cbar = fig.colorbar(c, cax=cax, label='PDF')
cbar.ax.tick_params(labelsize=14)
plt.savefig(r'figures/fig_8_energyFluxPDF.jpg', dpi=300, bbox_inches='tight')
plt.savefig(r'figuresFinal/fig_8_energyFluxPDF.png', dpi=300, bbox_inches='tight')
plt.savefig(r'figuresFinal/fig_8_energyFluxPDF.pdf', bbox_inches='tight')
plt.show()
