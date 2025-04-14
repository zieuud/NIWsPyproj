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


data1 = np.load(r'MoorData/ADCP_uv_ni_10bcmodes.npz')
ke_mod = data1['ke_mod']
ke_mod_data = np.nanmean(np.nanmean(ke_mod, -1), 0)
# data2 = np.load(r'MoorData/EnergyFlux_10bcmodes.npz')
# up_mod = data2['up_mod']
# vp_mod = data2['vp_mod']
# pp = np.load(r'MoorData/EnergyFlux.npz')['pp']
# pp_mod = pp.reshape(6650, 1, -1)
# fx_mod = pp_mod * up_mod
# fy_mod = pp_mod * vp_mod
data2 = np.load(r'MoorData/EnergyFlux_10bcmodes_fhProj_Ridge.npz')
fx_mod = data2['fx_mod']
fy_mod = data2['fy_mod']
f_mod = np.sqrt(fx_mod ** 2 + fy_mod ** 2)
f_mod_data = np.nanmean(np.nansum(f_mod, -1), 0)

ci_ke = np.empty(11)
for i in range(11):
    ci_ke[i] = cal_confidence(np.nanmean(ke_mod, -1))
ci_f = np.empty(11)
for i in range(11):
    ci_f[i] = cal_confidence(np.nansum(f_mod, -1))

plt.figure(1, figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.bar(range(11), ke_mod_data, color='#A6B1B7', yerr=ci_ke, capsize=6, error_kw=dict(ecolor='#444444', linewidth=1.5))
plt.xticks(range(11))
plt.ylim([0, 1200])
plt.ylabel('$KE_{NI}^{WKB}$ $J/m^{3}$')

plt.subplot(2, 1, 2)
plt.bar(range(11), f_mod_data, color='#A6B1B7', yerr=ci_f, capsize=6, error_kw=dict(ecolor='#444444', linewidth=1.5))
plt.xticks(range(11))
plt.ylim([0, 1200])
plt.xlabel('Modes')
plt.ylabel('$F_{NI}$ $W/m$')

# plt.savefig(r'figures/fig_7_ModeContribution.jpg', dpi=300, bbox_inches='tight')
plt.show()
