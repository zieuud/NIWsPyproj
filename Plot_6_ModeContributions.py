import numpy as np
import matplotlib.pyplot as plt
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


timeCut = 3960
nmodes = 10
data1 = np.load(r'MoorData/ADCP_uv_ni_10bcmodes.npz')
ke_mod = data1['ke_mod'][:timeCut, 1:, :]
ke_mod_data = np.nanmean(np.nansum(ke_mod, -1), 0)
data2 = np.load(r'MoorData/EnergyFlux_modes.npz')
fx_mod = data2['fx_mod'][:timeCut, :, :]
fy_mod = data2['fy_mod'][:timeCut, :, :]
f_mod = np.sqrt(fx_mod ** 2 + fy_mod ** 2)
f_mod_data = np.nanmean(np.nansum(f_mod, -1), 0)
data3 = np.load(r'MoorData/ADCP_uv.npz')
depthFlux = data3['depth_adcp'][:181]
# for checking
# f_mod[f_mod > 1e3] = np.nan
f_mod[f_mod == 0] = np.nan
# plt.pcolormesh(range(6650), depthFlux, f_mod[:, 0, :].T, vmin=0, vmax=100)
# plt.colorbar()
# plt.show()
ci_ke = np.empty(nmodes)
for i in range(nmodes):
    ci_ke[i] = cal_confidence(np.nansum(ke_mod, -1))
ci_f = np.empty(nmodes)
for i in range(nmodes):
    ci_f[i] = cal_confidence(np.nansum(f_mod, -1))

plt.figure(1, figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.bar(range(1, nmodes + 1), ke_mod_data, color='#A6B1B7', yerr=ci_ke, capsize=6, error_kw=dict(ecolor='#444444', linewidth=1.5))
plt.xticks(range(1, nmodes + 1))
plt.ylim([0, 80])
plt.ylabel('$KE_{NI}^{WKB}$ ($J/m^{2}$)')
plt.text(0.3, 0.95*80, 'a', ha='left', va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

plt.subplot(2, 1, 2)
plt.bar(range(1, nmodes + 1), f_mod_data, color='#A6B1B7', yerr=ci_f, capsize=6, error_kw=dict(ecolor='#444444', linewidth=1.5))
plt.xticks(range(1, nmodes + 1))
plt.ylim([0, 50])
plt.xlabel('Modes')
plt.ylabel('$F_{NI}$ ($W/m$)')
plt.text(0.3, 0.95*50, 'b', ha='left', va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

# plt.savefig(r'figures/fig_6_ModeContribution.jpg', dpi=300, bbox_inches='tight')
# plt.savefig(r'figuresFinal/fig_6_ModeContribution.png', dpi=300, bbox_inches='tight')
# plt.savefig(r'figuresFinal/fig_6_ModeContribution.pdf', bbox_inches='tight')
plt.show()
