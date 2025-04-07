import numpy as np

wind = np.load(r'ReanaData/ERA5_wind_moor.npz')
u10 = wind['u10']
v10 = wind['v10']

rhoAir = 1.3
dragCoeff = 1.5e-3
tao = rhoAir * dragCoeff * (u10 ** 2 + v10 ** 2)
tao_x = rhoAir * dragCoeff * u10 * np.sqrt(u10 ** 2 + v10 ** 2)
tao_y = rhoAir * dragCoeff * v10 * np.sqrt(u10 ** 2 + v10 ** 2)

# np.savez(r'ReanaData/ERA5_wind_stress.npz', tao=tao, tao_x=tao_x, tao_y=tao_y)

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

print('c')
