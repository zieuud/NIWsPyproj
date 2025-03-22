from datetime import datetime, timedelta
import numpy as np
from matplotlib import pyplot as plt

N2 = np.load(r'ReanaData/WOA23_N2_grid.npy')
moorData = np.load('ADCP_uv.npz')
depth = moorData['depth'][:180]
moorDate = moorData['mtime']
dates = [datetime(1, 1, 1) + timedelta(days=m-366) for m in moorDate]

uvp_ni = np.load(r'ReanaData\WOA23_uvp_ni.npz')
up_ni = uvp_ni['up_ni']
vp_ni = uvp_ni['vp_ni']

up_ni_wkb = np.copy(up_ni)
vp_ni_wkb = np.copy(vp_ni)
N2_averaged = np.nanmean(N2, 1)
max_idx = 180  # 最后一个N2非nan位置

for i in range(len(moorDate)):
    up_ni_wkb[i, :max_idx] = up_ni[i, :max_idx] * np.sqrt(np.sqrt(N2_averaged[i])/np.sqrt(N2[i, :max_idx]))
    vp_ni_wkb[i, :max_idx] = vp_ni[i, :max_idx] * np.sqrt(np.sqrt(N2_averaged[i])/np.sqrt(N2[i, :max_idx]))
KEp_ni_wkb = 1/2*1025*(up_ni_wkb**2+vp_ni_wkb**2)
np.savez(r'ReanaData\WOA23_uvp_ni_wkb.npz', up_ni_wkb=up_ni_wkb, vp_ni_wkb=vp_ni_wkb, KEp_ni_wkb=KEp_ni_wkb)