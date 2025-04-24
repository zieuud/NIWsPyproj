import numpy as np
import matplotlib.pyplot as plt


data1 = np.load(r'MoorData/ADCP_uv_ni_wkb.npz')
data2 = np.load(r'MoorData/ADCP_uv_ni_byYu.npz')

depth = data2['depth_adcp']
time = data2['mtime_adcp']
ke_ni_wkb = data1['KE_ni_wkb']
ke_ni = data2['ke_ni']
ke_ni_wkb_di = np.nansum(ke_ni_wkb, -1)
ke_ni_di = np.nansum(ke_ni, -1)

# check profile
plt.figure(1, figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.pcolormesh(time, depth, ke_ni.T, cmap='Oranges', vmin=0, vmax=15)
plt.colorbar()
plt.subplot(2, 1, 2)
plt.pcolormesh(time, depth, ke_ni_wkb.T, cmap='Oranges', vmin=0, vmax=15)
plt.colorbar()
# plt.savefig(r'figures/check_wkbscaling_profile.jpg', dpi=300, bbox_inches='tight')
plt.show()

# check conservation
plt.figure(2, figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(time, ke_ni_di, 'r-', label='$KE_{ni}$')
plt.plot(time, ke_ni_wkb_di, 'k-', label='$KE_{ni}^{wkb}$')
plt.legend()
plt.ylabel('$J/m^{2}$')
plt.subplot(2, 1, 2)
plt.plot(time, ke_ni_wkb_di - ke_ni_di, 'k-', label='($KE_{ni}^{wkb}$ - $KE_{ni}$)')
plt.axhline(0, color='r', linestyle='--')
plt.legend()
plt.ylabel('$J/m^{2}$')

# plt.savefig(r'figures/check_wkbscaling_conservation.jpg', dpi=300, bbox_inches='tight')
plt.show()
