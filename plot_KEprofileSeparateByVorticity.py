import numpy as np
import matplotlib.pyplot as plt


temp = np.load(r'ReanaData\WOA23_temp_grid.npy')
vorticity_moor = np.load(r'ReanaData\AVISO_vorticity1.npy')
adcp = np.load('ADCP_uv_ni_wkb.npz')
KE_ni = adcp['KE_ni_wkb']
adcp0 = np.load('ADCP_uv.npz')
moorDate = adcp0['mtime']
depth = adcp0['depth']
lat_moor = 36.23
fi = 2 * 7.292e-5 * np.sin(lat_moor/180*np.pi)
vf = vorticity_moor / fi
# --------- calculate KE_ni_wkb with negative and positive vorticity ----------
positiveVor = np.zeros(len(depth))
negativeVor = np.zeros(len(depth))
for idx in range(len(vorticity_moor)):
    if (vorticity_moor[idx] > 0) & (~np.isnan(KE_ni[idx, :]).all()):
        positiveVor += KE_ni[idx, :]
    elif (vorticity_moor[idx]) < 0 & (~np.isnan(KE_ni[idx, :]).all()):
        negativeVor += KE_ni[idx, :]
    elif np.isnan(KE_ni[idx, :]).all():
        print('nan value of KE_ni!')
    else:
        print('0 vorticity!')
# ---------- plot the KE_ni_wkb separate by vorticity ----------
plt.figure(1, figsize=(6, 8))
timeSpan = moorDate[-1] - moorDate[0]
plt.plot(positiveVor / timeSpan, depth, 'b-')
plt.plot(negativeVor / timeSpan, depth, 'r-')
plt.legend([r'$\zeta_{g} > 0$', r'$\zeta_{g} < 0$'])
plt.ylabel('depth (m)')
plt.xlabel('time-averaged $KE_{ni}^{wkb}$ ($J/m^{3}$)')
plt.savefig(r'figures\time-averaged KE separate by vorticity2.jpg', dpi=300)
plt.show()
print('c')
