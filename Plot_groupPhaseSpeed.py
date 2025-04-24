import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.fft import fft, ifft, fftfreq

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
# group speed line :
# event 1: cpz: 573, -93, 563, -247  cgz: 360, -35, 552, -176
# event 2: cpz: 2047, -138, 2038, -300  cgz: 2068, -63, 2285, -179
# event 3: cpz: 2936, -83, 2922, -222  cgz: 2868, -38, 2954, -139
# event 4: cpz: 4219, -29, 4200, -325  cgz: 4119, -32, 4269, -171
# event 5: cpz: 4886, -29, 4879, -160  cgz: 4818, -33, 4887, -80
# event 6: no obvious downward propagation
cpz_coord = [[573, -93, 563, -247], [2047, -138, 2038, -300], [2936, -83, 2922, -222],
             [4219, -29, 4200, -325], [4886, -29, 4879, -160]]
cgz_coord = [[360, -35, 552, -176], [2068, -63, 2285, -179], [2866, -59, 2932, -127],
             [4119, -32, 4269, -171], [4818, -33, 4887, -80]]
ee = 2
start = events[ee][0]
end = events[ee][1]
plt.figure(1)
for i in range(1, 122, 1):
    u_series = v_ni_wkb[start:end, i]
    l = depth[i]
    u_norm = 20 * (u_series - np.nanmin(u_series)) / (np.nanmax(u_series) - np.nanmin(u_series)) - 10
    plt.plot(range(start, end), u_norm + l, 'k-', linewidth=0.5)
plt.pcolormesh(range(start, end), depth[1:122], ke_ni_wkb[start:end, 1:122].T, vmin=0, vmax=10, cmap='Oranges')
plt.colorbar()
xp1, yp1, xp2, yp2 = cpz_coord[ee]
xg1, yg1, xg2, yg2 = cgz_coord[ee]
plt.plot([xp1, xp2], [yp1, yp2], 'g-')
cpz = (yp2 - yp1) / (xp2 - xp1) / 3600
plt.text(xp2, yp2, '$c_{{\phi z }}=${:.2e} m/s'.format(cpz))
plt.plot([xg1, xg2], [yg1, yg2], 'g-')
cgz = (yg2 - yg1) / (xg2 - xg1) / 3600
plt.text(xg2, yg2, '$c_{{g z }}=${:.2e} m/s'.format(cgz))
# plt.show()

lat_moor = 36.23
fi = 2 * 7.292e-5 * np.sin(np.deg2rad(lat_moor))
N0 = 3.6e-3
cpz5 = []
cgz5 = []
m5 = []
omega5 = []
kh5 = []
for i in range(5):
    xp1, yp1, xp2, yp2 = cpz_coord[i]
    xg1, yg1, xg2, yg2 = cgz_coord[i]
    cpz = (yp2 - yp1) / (xp2 - xp1) / 3600
    cgz = (yg2 - yg1) / (xg2 - xg1) / 3600
    a = cpz ** 2
    b = -2 * cgz * fi
    c = -fi ** 2
    m = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    a = 1
    b = -cgz * m
    c = -fi ** 2
    omega = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    kh = np.sqrt((omega ** 2 - fi ** 2) * m ** 2 / N0 ** 2)
    cpz5.append(cpz)
    cgz5.append(cgz)
    m5.append(m)
    omega5.append(omega / fi)
    kh5.append(kh)

plt.figure(2)
plt.subplot(1, 2, 1)
m_mean = np.mean(m5)
m_std = np.std(m5)
cgz5_mean = np.mean(np.abs(cgz5))
cgz5_std = np.std(np.abs(cgz5))
cgz_dispersion = [[1, 1.1], [0, np.abs(((1.1 * fi) ** 2 - fi ** 2) / (1.1 * fi * m_mean))]]
cgz_dispersion_up = [[1, 1.1], [0, np.abs(((1.1 * fi) ** 2 - fi ** 2) / (1.1 * fi * (m_mean - m_std)))]]
cgz_dispersion_low = [[1, 1.1], [0, np.abs(((1.1 * fi) ** 2 - fi ** 2) / (1.1 * fi * (m_mean + m_std)))]]
plt.plot(cgz_dispersion[0], cgz_dispersion[1], 'k-')
plt.plot(cgz_dispersion_up[0], cgz_dispersion_up[1], 'k--')
plt.plot(cgz_dispersion_low[0], cgz_dispersion_low[1], 'k--')
plt.scatter(omega5, np.abs(cgz5))
plt.axhline(cgz5_mean, color='b', linestyle='--')
plt.axhline(cgz5_mean - cgz5_std, color='b', linestyle='--')
plt.axhline(cgz5_mean + cgz5_std, color='b', linestyle='--')

plt.subplot(1, 2, 2)
m_mean = np.abs(np.mean(m5))
m_std = np.abs(np.std(m5))
cpz5_mean = np.mean(cpz5)
cpz5_std = np.std(cpz5)
cpz_dispersion = [[1, 1.1], [0, 1.1 * fi / m_mean]]
cpz_dispersion_up = [[1, 1.1], [0, 1.1 * fi / (m_mean - m_std)]]
cpz_dispersion_low = [[1, 1.1], [0, 1.1 * fi / (m_mean + m_std)]]
plt.plot(cpz_dispersion[0], cpz_dispersion[1], 'k-')
plt.plot(cpz_dispersion_up[0], cpz_dispersion_up[1], 'k--')
plt.plot(cpz_dispersion_low[0], cpz_dispersion_low[1], 'k--')
plt.scatter(omega5, cpz5)
plt.axhline(cpz5_mean, color='b', linestyle='--')
plt.axhline(cpz5_mean - cpz5_std, color='b', linestyle='--')
plt.axhline(cpz5_mean + cpz5_std, color='b', linestyle='--')

plt.show()

print('c')
# 2068 -63
# 2285 -179
# vorticity
# vor = np.load(r'ReanaData/AVISO_0125_vorticity3.npy')
# plt.figure()
# plt.subplot(211)
# plt.plot(range(6650), vor)
# plt.axhline(0, color='r', linestyle='--')
# plt.subplot(212)
# plt.pcolormesh(range(6650), depth, ke_ni_wkb.T, vmin=0, vmax=15)
# plt.show()
# idx = [[1, 3], [2, 4]]
# plt.figure(1, figsize=(10, 6))
# for i in range(2, 4):
#     start = events[i][0]
#     end = events[i][1]
#     plt.subplot(2, 2, idx[i-2][0])
#     plt.pcolormesh(time[start:end], depth[:idx1000], v_ni_wkb[start:end, :idx1000].T, cmap='seismic', vmin=-0.3, vmax=0.3)
#     plt.subplot(2, 2, idx[i-2][1])
#     plt.pcolormesh(time[start:end], depth[:idx1000], ke_ni_wkb[start:end, :idx1000].T, cmap='OrRd', vmin=0, vmax=15)
# plt.savefig(r'figures/check2.jpg', dpi=600, bbox_inches='tight')
# plt.colorbar()
# plt.show()
