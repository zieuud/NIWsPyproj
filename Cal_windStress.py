import matplotlib.pyplot as plt
import numpy as np
from func_0_filter import filter_ni
import scipy.signal as sig


wind = np.load(r'ReanaData/ERA5_wind_moor.npz')
u10 = wind['u10']
v10 = wind['v10']

# ---------- calculate wind stress ----------
rhoAir = 1.3
dragCoeff = 1.5e-3
tao = rhoAir * dragCoeff * (u10 ** 2 + v10 ** 2)
tao_x = rhoAir * dragCoeff * u10 * np.sqrt(u10 ** 2 + v10 ** 2)
tao_y = rhoAir * dragCoeff * v10 * np.sqrt(u10 ** 2 + v10 ** 2)
# ---------- slab model ----------
fi = 2 * 7.292e-5 * np.sin(np.deg2rad(36.25))  # inertial frequency rad/s
r = 0.08 * fi  # damping parameter
omega = r + 1j * fi  # complex frequency
T = (tao_x + 1j * tao_y) / 1025  # complex wind stress
dt = 3600
Z = np.zeros(6650, dtype=complex)
Z[0] = 0 + 1j * 0
for i in range(1, 6650):
    T_rate = (T[i] - T[i-1])/dt
    Z[i] = Z[i-1] * np.exp(-omega * dt) - T_rate * (1 - np.exp(-omega * dt)) / (omega ** 2 * 75)
u_slab = Z.real
v_slab = Z.imag
# ---------- calculate the near inertial wind work with observed data and slab model data ----------
uv_ni = np.load(r'MoorData/ADCP_uv_ni_byYu.npz')
u_ni = uv_ni['u_ni']
v_ni = uv_ni['v_ni']

tao_x_ni = np.squeeze(filter_ni(tao_x.reshape(-1, 1), 3600, 6650, 36.25))
tao_y_ni = np.squeeze(filter_ni(tao_y.reshape(-1, 1), 3600, 6650, 36.25))
windWork_ni_obs = np.nanmean(u_ni[:, :1], 1) * tao_x_ni + np.nanmean(v_ni[:, :1], 1) * tao_y_ni
windWork_ni_slab = u_slab * tao_x_ni + v_slab * tao_y_ni
windWork_ni_obs_cum = 3600 * np.nancumsum(windWork_ni_obs) / 1000
windWork_ni_slab_cum = 3600 * np.nancumsum(windWork_ni_slab) / 1000

np.savez(r'ReanaData/ERA5_tao_ni.npz', tao_x_ni=tao_x_ni, tao_y_ni=tao_y_ni)
np.savez(r'ReanaData/ERA5_windWork.npz', windWork_ni_obs=windWork_ni_obs, windWork_ni_slab=windWork_ni_slab)
np.savez(r'ReanaData/ERA5_windWork_cum.npz', windWork_ni_obs_cum=windWork_ni_obs_cum, windWork_ni_slab_cum=windWork_ni_slab_cum)

# for checking
plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(range(6650), tao, 'k-', label='ECMWF')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(range(6650), windWork_ni_obs, 'r-', label='observed')
plt.plot(range(6650), windWork_ni_slab, 'k-', label='slab model')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(range(6650), windWork_ni_obs_cum, 'r-', label='observed')
plt.plot(range(6650), windWork_ni_slab_cum, 'k-', label='slab model')
plt.legend()
plt.show()

print('c')
