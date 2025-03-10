import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


N2 = np.load(r'ReanaData\WOA23_N2_grid.npy')
adcp = np.load('ADCP_uv_ni_wkb.npz')
KE_ni = adcp['KE_ni_wkb']
adcp0 = np.load('ADCP_uv.npz')
moorDate = adcp0['mtime']
dateForPlot = [datetime(1, 1, 1) + timedelta(days=m-366) for m in moorDate]
depth = adcp0['depth']
woa23Temp = np.load(r'ReanaData\WOA23_temp_grid.npy')
# ---------- find the depth of mixed layer ----------
ml_depthArray = []
for idx in range(np.size(woa23Temp, 0)):
    ml_idx = np.argwhere(woa23Temp[idx, :] - woa23Temp[idx, 0] < -0.5)[0]
    ml_depthArray.append(depth[ml_idx])
ml_depthArray = np.array(ml_depthArray)
# --------- plot KE_ni_wkb profile ----------
plt.figure(1, figsize=(10, 6))
[depth_mesh, moorDate_mesh] = np.meshgrid(depth[:], dateForPlot)
c = plt.pcolor(moorDate_mesh, depth_mesh, KE_ni[:, :], cmap='Oranges', vmin=0, vmax=10)
cb = plt.colorbar(c)
cb.set_label(r'$KE_{NI}^{WKB}$ $(J/m^{3})$')
plt.ylabel('depth (m)')
line, = plt.plot(dateForPlot, ml_depthArray, label='$H_{ML}$')
plt.legend(handles=[line])
plt.savefig(r'figures\KE_ni_wkb_profile_with_pycnocline.jpg', dpi=300)
# plt.show()
