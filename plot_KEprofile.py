import numpy as np
import matplotlib.pyplot as plt

N2 = np.load(r'ReanaData\WOA23_N2_grid.npy')
adcp = np.load('ADCP_uv_ni_wkb.npz')
KE_ni = adcp['KE_ni_wkb']

adcp0 = np.load('ADCP_uv.npz')
moorDate = adcp0['mtime']
depth = adcp0['depth']

woa23 = np.load(r'ReanaData\WOA23_st.npz')
s = woa23['s']
t = woa23['t']
z = -woa23['z']
[lat, lon] = woa23['loc']

for i in range(12):
    dt = t[i, 1:] - t[i, :-1]
    idx = np.argmax(abs(dt))
    plt.plot(t[i, :])
    plt.scatter(idx, t[i, idx])
    plt.show()


# [depth_mesh, moorDate_mesh] = np.meshgrid(depth[:120], moorDate)
# c = plt.pcolor(moorDate_mesh, depth_mesh, KE_ni[:, :120], cmap='Oranges', vmin=0, vmax=10)
# plt.plot(moorDate, ml_idxArray)
# plt.show()
