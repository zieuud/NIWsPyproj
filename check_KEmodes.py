import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

adcp_modes = np.load(r'MoorData/ADCP_uv_ni_10bcmodes_test.npz')
KE_mod = adcp_modes['ke_mod']
adcp0 = np.load('MoorData/ADCP_uv.npz')
moorDate = adcp0['mtime_adcp']
dateForPlot = [datetime(1, 1, 1) + timedelta(days=m - 367) for m in moorDate]
depth = adcp0['depth_adcp']
# ---------- multi-depth KE modes profile ----------
[depth_mesh, moorDate_mesh] = np.meshgrid(depth[:180], dateForPlot)
plt.figure(1, figsize=(10, 10))
for i in range(1, 11):
    plt.subplot(5, 2, i)
    c = plt.pcolormesh(moorDate, depth, np.squeeze(KE_mod[:, i, :].T), cmap='Oranges', vmin=0, vmax=10)
    cb = plt.colorbar(c)
    cb.set_label(r'$KE_{NI}^{WKB}$ $(J/m^{3})$')
# plt.savefig(r'figures\KE_profile_modes.jpg', dpi=300)
plt.show()
# ---------- time-averaged modes KE profile ----------
# plt.figure(2, figsize=(6, 8))
# for i in range(5):
#     plt.plot(np.nanmean(np.squeeze(KE_mod[:, i, :]), 0), depth[:])
# plt.xlabel(r'time-averaged $KE_{NI}^{WKB}$ $(J/m^{3})$')
# plt.ylabel(r'depth (m)')
# plt.title(r'time-averaged $KE_{ni}^{wkb}$ of 5 modes')
# plt.legend(['mode 0', 'mode 1', 'mode 2', 'mode 3', 'mode >= 4'])
# plt.savefig(r'figures\KE modes.jpg', dpi=300)
# plt.show()
# ---------- time-averaged depth-integrated modes KE profile
# KE_mod_dita = np.trapz(np.squeeze(np.nanmean(KE_mod, 0)), depth[:])
# dz = depth[1] - depth[0]
# KE_mod_dita = np.nanmean(np.nanmean(KE_mod[:, :, :], -1), 0)
# plt.figure(3)
# plt.bar(range(11), KE_mod_dita)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# plt.xlabel('modes')
# plt.ylabel('$KE_{ni}^{wkb}$ $(J·m^{-3})$')
# plt.savefig(r'figures/time-averaged depth-integrated modes NIKE.jpg', dpi=300)
# plt.savefig(r'figures/check_KE_mdoes_way_c.jpg', dpi=300)
plt.show()
