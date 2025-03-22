import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


adcp = np.load('ADCP_uv_ni_wkb.npz')
KE_ni = adcp['KE_ni_wkb']
adcp0 = np.load('ADCP_uv.npz')
moorDate = adcp0['mtime']
depth = adcp0['depth']
dateForPlot = [datetime(1, 1, 1) + timedelta(days=m-366) for m in moorDate]
moorPycnocline = -np.load(r'ReanaData\GLORYS_pycnocline.npy')

# --------- plot KE_ni_wkb profile ----------
plt.figure(1, figsize=(10, 6))
[depth_mesh, moorDate_mesh] = np.meshgrid(depth, dateForPlot)
c = plt.pcolor(moorDate_mesh, depth_mesh, KE_ni, cmap='Oranges', vmin=0, vmax=10)
cb = plt.colorbar(c)
cb.set_label(r'$KE_{NI}^{WKB}$ $(J/m^{3})$')
plt.ylabel('depth (m)')
line, = plt.plot(dateForPlot, moorPycnocline, label='$H_{ML}$')
plt.legend(handles=[line])
plt.savefig(r'figures\KE_ni_wkb profile with pycnocline.jpg', dpi=300)
# plt.show()
