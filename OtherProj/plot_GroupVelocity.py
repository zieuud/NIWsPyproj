import numpy as np
from matplotlib import pyplot as plt
import gsw
from scipy.io import loadmat

uv_mode = np.load('ParallelLine4_uv_filter_mod.npz')
# --- plot mode NIKE
u = uv_mode['u_pline_filter_mod'][:]
v = uv_mode['v_pline_filter_mod'][:]
u_bt_typ = np.mean(u[:, :, 0, 408:456], (1, 2))
v_bt_typ = np.mean(v[:, :, 0, 408:456], (1, 2))
NIKE_bt_typ = 1/2*1025*(u_bt_typ**2+v_bt_typ**2)

u_bc1_typ = np.mean(u[:, :, 1, 408:456], (1, 2))
v_bc1_typ = np.mean(v[:, :, 1, 408:456], (1, 2))
NIKE_bc1_typ = 1/2*1025*(u_bc1_typ**2+v_bc1_typ**2)

u_bc2_typ = np.mean(np.sum(u[:, :, 2:, 408:456], 2), (1, 2))
v_bc2_typ = np.mean(np.sum(v[:, :, 2:, 408:456], 2), (1, 2))
NIKE_bc2_typ = 1/2*1025*(u_bc2_typ**2+v_bc2_typ**2)

u_bt_ptyp = np.mean(u[:, :, 0, 456:600], (1, 2))
v_bt_ptyp = np.mean(v[:, :, 0, 456:600], (1, 2))
NIKE_bt_ptyp = 1/2*1025*(u_bt_ptyp**2+v_bt_ptyp**2)

u_bc1_ptyp = np.mean(u[:, :, 1, 456:600], (1, 2))
v_bc1_ptyp = np.mean(v[:, :, 1, 456:600], (1, 2))
NIKE_bc1_ptyp = 1/2*1025*(u_bc1_ptyp**2+v_bc1_ptyp**2)

u_bc2_ptyp = np.mean(np.sum(u[:, :, 2:, 456:600], 2), (1, 2))
v_bc2_ptyp = np.mean(np.sum(v[:, :, 2:, 456:600], 2), (1, 2))
NIKE_bc2_ptyp = 1/2*1025*(u_bc2_ptyp**2+v_bc2_ptyp**2)

plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(range(352), NIKE_bt_typ)
plt.plot(range(352), NIKE_bc1_typ)
plt.plot(range(352), NIKE_bc2_typ)

plt.subplot(2, 1, 2)
plt.plot(range(352), NIKE_bt_ptyp)
plt.plot(range(352), NIKE_bc1_ptyp)
plt.plot(range(352), NIKE_bc2_ptyp)

plt.show()

# --- plot Group Velocity
modec = np.load('mode_decom.npz')
e = modec['e']
coord = loadmat(r'L:\NIWs\ExtractData\CaseV5\ParallelLine4_coord.mat')
lat = coord['lat_line'][:, :]
f = gsw.f(lat)
omega1 = 0.8*f



