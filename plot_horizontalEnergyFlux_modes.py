import matplotlib.pyplot as plt
import numpy as np


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

F_modes = np.sqrt(fx_modes ** 2 + fy_modes ** 2)
# F_modes_di = np.nansum(F_modes, 2) * dz
# plt.figure(1)
# plt.plot(F_modes_di)
# plt.legend(['mode{}'.format(i) for i in range(0, 11)])
# plt.show()

F_modes_dita = np.nanmean(np.nansum(F_modes, 2) * dz, 0)
# F_modes_dita = np.append(F_modes_dita, np.nansum(F_modes_dita[2:]))

plt.figure(2)
plt.bar(range(0, 11), F_modes_dita)
plt.xticks(range(0, 11), ['{}'.format(i) for i in range(0, 11)])
plt.xlabel('Modes')
plt.ylabel('Energy Flux ($W·m^{-1}$)')
plt.savefig(r'figures\Time-averaged Depth-integrated Energy Flux modes.jpg', dpi=300)
plt.show()
print('c')
