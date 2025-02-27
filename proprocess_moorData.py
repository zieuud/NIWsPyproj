from scipy.io import loadmat
import numpy as np
from datetime import datetime, timedelta


data = loadmat('uv_ADCP.mat')
depth = -np.squeeze(data['depth_ADCP']).astype(float)
mtime = np.squeeze(data['mtime_ADCP'])
u = data['u_ADCP']
v = data['v_ADCP']
date = [datetime(1, 1, 1) + timedelta(days=m-366) for m in mtime]
np.savez('ADCP_uv.npz', u=u, v=v, depth=depth, mtime=mtime)
