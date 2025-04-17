import numpy as np
import matplotlib.pyplot as plt


data1 = np.load(r'MoorData/ADCP_uv_ni_wkb.npz')
u_ni_wkb = data1['u_ni_wkb']
v_ni_wkb = data1['v_ni_wkb']
ke_ni_wkb = data1['KE_ni_wkb']
data2 = np.load(r'MoorData/ADCP_uv.npz')
depth = data2['depth_adcp']
time = data2['mtime_adcp']
idx1000 = 122

events = [[1, 1200], [1600, 2650], [2800, 3080], [4000, 4300], [4750, 4900], [6080, 6300]]
numEvents = 2

plt.subplot(2, 1, 1)
plt.pcolormesh(time[1:1200], depth[:idx1000], v_ni_wkb[1:1200, :idx1000].T, cmap='seismic', vmin=-0.3, vmax=0.3)
plt.plot([7.362e5+51, 7.362e5+60], [-50, -230], 'k-')
plt.subplot(2, 1, 2)
plt.pcolormesh(time[1:1200], depth[:idx1000], ke_ni_wkb[1:1200, :idx1000].T, cmap='OrRd', vmin=0, vmax=15)
plt.plot([7.362e5+51, 7.362e5+60], [-50, -230], 'k-')
# plt.colorbar()
plt.show()
