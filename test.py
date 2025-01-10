from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

data = loadmat('uv_ADCP.mat')
print(data.keys())
depth = -np.squeeze(data['depth_ADCP']).astype(float)
mtime = np.squeeze(data['mtime_ADCP'])
u = data['u_ADCP']
v = data['v_ADCP']

jd_start = datetime(4713, 1, 1)  # 儒略日的起始时间
# 转换为标准日期时间
dates = [datetime(4713, 1, 1) + timedelta(days=m - 1721424.5) for m in mtime]
print(dates)
print('test')
# np.savez('ADCP_uv.npz', u=u, v=v, depth=depth, mtime=mtime)
# [time_axis, depth_axis] = np.meshgrid(mtime, depth)
#
# plt.pcolor(time_axis, depth_axis, u)
# plt.show()
