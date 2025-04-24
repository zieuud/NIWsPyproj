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
# # calculate the r error
# ci = cal_confidence(fh) * 10  # enlarge 10 times also for consistency
# r_low = r_arrow - ci
# r_high = r_arrow + ci
# calculate the theta error
varianceRatio = pca.explained_variance_ratio_[0]
width = 1.96 * np.sqrt(pca.explained_variance_[0] / len(fx))  # 1.96 to estimate the 95% confidence interval
spread_angle = np.arctan2(width, 1)
theta_left = theta_arrow - spread_angle
theta_right = theta_arrow + spread_angle

# ---------- plot PDF ----------
hist, theta_edges, r_edges = np.histogram2d(theta, magnitude, bins=(36, 20))
hist_density = (hist - np.nanmin(hist)) / (np.nanmax(hist) - np.nanmin(hist))
theta_mesh, magnitude_mesh = np.meshgrid(theta_edges, r_edges)
plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
c = ax.pcolormesh(theta_mesh, magnitude_mesh, hist_density.T, cmap='OrRd', norm=LogNorm(vmin=5e-3, vmax=1))
ax.set_yticks([200, 400, 600])
plt.colorbar(c, label='PDF', orientation='horizontal')
# add the arrow of main axis
ax.annotate('', xy=(theta_arrow, r_arrow), xytext=(0, 0),
            arrowprops=dict(facecolor='#0074e4', edgecolor='#0074e4', width=2, headwidth=5),
            annotation_clip=False, zorder=2)
# add the error on theta
ax.plot([theta_left, theta_left],
        [0, r_arrow], linestyle='--', color='#0074e4', linewidth=1.5)
ax.plot([theta_right, theta_right],
        [0, r_arrow], linestyle='--', color='#0074e4', linewidth=1.5)
# # add the error on r
# ax.plot([theta_arrow - np.radians(5), theta_arrow + np.radians(5)],
#         [r_high, r_high], color='#666666', linewidth=1.5, zorder=3)
# ax.plot([theta_arrow - np.radians(5), theta_arrow + np.radians(5)],
#         [r_low, r_low], color='#666666', linewidth=1.5, zorder=3)

plt.title('PDF of Horizontal Energy Flux')

# plt.savefig(r'figures\fig_8_1_EnergyFluxPDF.jpg', dpi=300, bbox_inches='tight')
plt.show()
print('c')
