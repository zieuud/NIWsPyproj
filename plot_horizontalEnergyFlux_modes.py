import matplotlib.pyplot as plt
import numpy as np


data = np.load(r'MoorData/EnergyFlux_10bcmodes.npz')
up_mod = data['up_mod']
vp_mod = data['vp_mod']
pp_mod = data['vp_mod']
fx_mod = pp_mod * up_mod
fy_mod = pp_mod * vp_mod
nt, nmodes, nz = np.shape(up_mod)
dz = 8
dt = 3600

F_modes = np.sqrt(fx_mod ** 2 + fy_mod ** 2)
# F_modes_di = np.nansum(F_modes, 2) * dz
# plt.figure(1)
# plt.plot(F_modes_di)
# plt.legend(['mode{}'.format(i) for i in range(0, 11)])
# plt.show()

F_modes_dita = np.nanmean(np.nanmean(F_modes, 2) * dz, 0)
# F_modes_dita = np.append(F_modes_dita, np.nansum(F_modes_dita[2:]))

plt.figure(2)
plt.bar(range(0, 11), F_modes_dita)
plt.xticks(range(0, 11), ['{}'.format(i) for i in range(0, 11)])
plt.xlabel('Modes')
plt.ylabel('Energy Flux ($W·m^{-1}$)')
# plt.savefig(r'figures\Time-averaged Depth-integrated Energy Flux modes.jpg', dpi=300)
plt.show()
print('c')
