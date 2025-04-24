import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
from matplotlib.ticker import ScalarFormatter


def cal_prop_params(cpz_coord, cgz_coord):
    lat_moor = 36.23
    fi = 2 * 7.292e-5 * np.sin(np.deg2rad(lat_moor))
    N0 = 3.6e-3
    cpz5 = []
    cgz5 = []
    m5 = []
    omega5 = []
    kh5 = []
    for i in range(5):
        xp1, yp1, xp2, yp2 = cpz_coord[i]
        xg1, yg1, xg2, yg2 = cgz_coord[i]
        cpz = (yp2 - yp1) / (xp2 - xp1) / 3600
        cgz = (yg2 - yg1) / (xg2 - xg1) / 3600
        m = cal_square_equation(cpz ** 2, -2 * cgz * fi, -fi ** 2)[-1]
        omega = cal_square_equation(1, -cgz * m, -fi ** 2)[0]
        kh = np.sqrt((omega ** 2 - fi ** 2) * m ** 2 / N0 ** 2)
        cpz5.append(cpz)
        cgz5.append(cgz)
        m5.append(m)
        omega5.append(omega / fi)
        kh5.append(kh)
    return np.array(cpz5), np.array(cgz5), np.array(m5), np.array(omega5), np.array(kh5)


def cal_square_equation(a, b, c):
    return (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a), (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)


data1 = np.load(r'MoorData/ADCP_uv_ni_wkb.npz')
u_ni_wkb = data1['u_ni_wkb']
v_ni_wkb = data1['v_ni_wkb']
ke_ni_wkb = data1['KE_ni_wkb']
data2 = np.load(r'MoorData/ADCP_uv.npz')
depth = data2['depth_adcp']
time = data2['mtime_adcp']
date = [datetime(1, 1, 1) + timedelta(days=m - 367) for m in time]
# hours since 2015-09-27 05:00:00
idx1000 = 40
events = [[0, 1200], [1600, 2650], [2800, 3080], [4000, 4300], [4750, 4900], [6080, 6300]]
cpz_coord = [[573, -93, 563, -247], [2047, -138, 2038, -300], [2936, -83, 2922, -222],
             [4219, -29, 4200, -325], [4886, -29, 4879, -160]]
cgz_coord = [[360, -35, 552, -176], [2068, -63, 2285, -179], [2866, -59, 2932, -127],
             [4119, -32, 4269, -171], [4818, -33, 4887, -80]]

fig = plt.figure(1, figsize=(15, 10))
gs = gridspec.GridSpec(3, 4, width_ratios=[8, 2, 2, 1], height_ratios=[1, 50, 50], hspace=0.2, wspace=0.2)
subLoc = [4, 8, 5, 6, 7]
subIdx = ['a event 1', 'e event 2', 'b event 3', 'c event 4', 'd event 5']
for i in range(5):
    ax = fig.add_subplot(gs[subLoc[i]])
    start, end = events[i]
    for j in range(1, idx1000):
        u_series = v_ni_wkb[start:end, j]
        u_norm = 20 * (u_series - np.nanmin(u_series)) / (np.nanmax(u_series) - np.nanmin(u_series)) - 10
        ax.plot(range(start, end), u_norm + depth[j], 'k-', linewidth=0.5)
    pcm = ax.pcolormesh(range(start, end), depth[:idx1000], ke_ni_wkb[start:end, :idx1000].T, vmin=0, vmax=16,
                        cmap='Oranges')
    xp1, yp1, xp2, yp2 = cpz_coord[i]
    xg1, yg1, xg2, yg2 = cgz_coord[i]
    ax.plot([xp1, xp2], [yp1, yp2], color='b', linestyle='-')
    cpz = (yp2 - yp1) / (xp2 - xp1) / 3600
    ax.text(start, 5, '$c_{{\phi z }}=${:.2e} m/s'.format(cpz), color='b', fontsize=8, fontweight='bold', ha='left',
            va='top')
    ax.plot([xg1, xg2], [yg1, yg2], color='g', linestyle='-')
    cgz = (yg2 - yg1) / (xg2 - xg1) / 3600
    ax.text(start, -5, '$c_{{g z }}=${:.2e} m/s'.format(cgz), color='g', fontsize=8, fontweight='bold', ha='left',
            va='top')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    if i > 1:
        ax.set_yticks([])
    else:
        ax.set_ylabel('depth (m)')
    if i == 1:
        ax.set_xlabel('hours since 2015-09-27 05:00:00')
    ax.set_ylim([depth[idx1000], 5])
    ax.text(start, depth[idx1000], subIdx[i], ha='left', va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

cax = fig.add_subplot(gs[0, :])
cbar = fig.colorbar(pcm, cax=cax, orientation='horizontal')
cbar.set_label(r'$KE_{NI}^{WKB}$ $(J/m^{3})$')
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.tick_top()

ax = fig.add_subplot(gs[2, 1:])
lat_moor = 36.23
fi = 2 * 7.292e-5 * np.sin(np.deg2rad(lat_moor))
cpz5, cgz5, m5, omega5, kh5 = cal_prop_params(cpz_coord, cgz_coord)
m_mean = np.mean(m5)
m_std = np.std(m5)
cgz5_mean = np.mean(np.abs(cgz5))
cgz5_std = np.std(np.abs(cgz5))
cgz_dispersion = np.array([[1, 1.1], [0, np.abs(((1.1 * fi) ** 2 - fi ** 2) / (1.1 * fi * m_mean))]])
cgz_dispersion_up = np.array([[1, 1.1], [0, np.abs(((1.1 * fi) ** 2 - fi ** 2) / (1.1 * fi * (m_mean - m_std)))]])
cgz_dispersion_low = np.array([[1, 1.1], [0, np.abs(((1.1 * fi) ** 2 - fi ** 2) / (1.1 * fi * (m_mean + m_std)))]])
ax.plot(cgz_dispersion[0], cgz_dispersion[1] * 1e4, 'g-')
ax.plot(cgz_dispersion_up[0], cgz_dispersion_up[1] * 1e4, 'g--')
ax.plot(cgz_dispersion_low[0], cgz_dispersion_low[1] * 1e4, 'g--')
ax.fill_between(cgz_dispersion_up[0], cgz_dispersion_low[1] * 1e4, cgz_dispersion_up[1] * 1e4,
                color='#A8D5BA', alpha=0.5, edgecolor='none')
colors = ['#FFB3B3',  # 比原来的 #FFCCCC 深一些（浅粉红）
          '#FF8080',  # 比 #FF9999 深（中亮粉）
          '#FF4D4D',  # 比 #FF6666 深（正红）
          '#B32424',  # 深红
          '#801515']  # 更深棕红
label = ['event {}'.format(i) for i in range(1, 6)]
for a, b, c, l in zip(omega5, np.abs(cgz5 * 1e4), colors, label):
    ax.scatter(a, b, color=c, label=l)
ax.axhline(cgz5_mean * 1e4, color='b', linestyle='-')
ax.axhline((cgz5_mean - cgz5_std) * 1e4, color='b', linestyle='--')
ax.axhline((cgz5_mean + cgz5_std) * 1e4, color='b', linestyle='--')
ax.axhspan((cgz5_mean - cgz5_std) * 1e4, (cgz5_mean + cgz5_std) * 1e4,
           facecolor='#A7C7E7', alpha=0.5, edgecolor='none')
ax.text(1.001, 0.1, 'f', ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
ax.set_ylim([0, 3.5])
ax.set_xlim([1, 1.08])
ax.set_ylabel('$c_{gz}$ ($10^{-4}$ m/s)', labelpad=-2)
ax.set_xlabel('$\\omega/f$')
ax.set_yticks([0, 1, 2, 3])
ax.legend()

plt.savefig(r'figures/fig_7_GroupPhaseVelocity.jpg', dpi=300, bbox_inches='tight')
plt.savefig(r'figuresFinal/fig_7_GroupPhaseVelocity.png', dpi=300, bbox_inches='tight')
plt.savefig(r'figuresFinal/fig_7_GroupPhaseVelocity.pdf', bbox_inches='tight')
plt.show()
