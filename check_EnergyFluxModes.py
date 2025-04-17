import matplotlib.pyplot as plt
import numpy as np


endIdx1 = 181
endIdx2 = 172
adcp = np.load(r'MoorData/ADCP_uv.npz')
depth1 = adcp['depth_adcp'][9:endIdx1]
depth2 = adcp['depth_adcp'][9:endIdx2]
depth3 = adcp['depth_adcp'][:endIdx1]

fx_mod = np.load(r'MoorData/EnergyFlux_10bcmodes_fhProj_1400m.npz')['fx_mod']
fy_mod = np.load(r'MoorData/EnergyFlux_10bcmodes_fhProj_1400m.npz')['fy_mod']
fh_mod = np.sqrt(fx_mod ** 2 + fy_mod ** 2)

fx = np.load(r'MoorData/EnergyFlux.npz')['fx']
fy = np.load(r'MoorData/EnergyFlux.npz')['fy']
fh = np.sqrt(fx ** 2 + fy ** 2)


plt.figure(1, figsize=(10, 6))
for m in range(6):
    if m == 0:
        plt.subplot(6, 1, 1)
        plt.pcolormesh(range(6650), depth1, fh.T, cmap='Oranges', vmin=0, vmax=1e1)
        plt.colorbar()
        plt.xticks([])
    else:
        plt.subplot(6, 1, m+1)
        plt.pcolormesh(range(6650), depth2, fh_mod[:, m - 1, :].T, cmap='Oranges', vmin=0, vmax=1e1)
        plt.colorbar()
        plt.xticks([])

plt.savefig(r'figures/check_energyFluxModes_fhDecom_1400m.jpg', dpi=300, bbox_inches='tight')
plt.show()
# fh_mod_di = np.nansum(fh_mod, -1)
# fh_di = np.nansum(fh, -1)

# for i in range(10):
#     plt.plot(range(6650), fh_mod_di[:, i], label='mode {}'.format(i+1))
# plt.plot(range(6650), fh_di, label='total')
# plt.legend()
# plt.show()
