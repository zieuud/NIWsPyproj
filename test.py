import numpy as np
import matplotlib.pyplot as plt
from spectrum import pmtm
from scipy.signal import windows


def cal_spectrum(u, v, dt):
    uv_complex = u + 1j * v
    uv_complex = np.nan_to_num(uv_complex)  # Replace NaN with zero
    e, f, _ = pmtm(uv_complex, NW=3, return_freq=True)
    spp = np.abs(e)**2  # Clockwise spectrum
    snn = np.abs(e.conj())**2  # Counter-clockwise spectrum
    return f.real.T * dt, spp, snn


adcp = np.load(r'ADCP_uv.npz')
u = adcp['u']
v = adcp['v']
depth = adcp['depth']
time = adcp['mtime']
dt = 3600
dx = depth[1] - depth[0]
lat_moor = 36.23
fi = 2 * 7.292e-5 * np.sin(np.deg2rad(lat_moor)) / (2 * np.pi)
Nx = len(depth)
Nt = len(time)

# 构造二维波信号 (示例：空间和时间上的正弦波叠加)
X, T = np.meshgrid(depth, time)


# 2D FFT（二维快速傅里叶变换）
u_hat = np.fft.fft2(u)
u_hat_shifted = np.fft.fftshift(u_hat)  # 将零频移到中心

# 计算波数和频率
kx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))  # 波数
omega = np.fft.fftshift(np.fft.fftfreq(Nt, d=dt))  # 频率

# 计算波数-频率谱 (归一化)
E_k_w = np.abs(u_hat_shifted)**2 / (Nx * Nt)
kx, omega = np.meshgrid(kx, omega)
# 绘图
plt.figure(figsize=(8, 6))
plt.pcolor(kx.T, omega.T, np.log10(E_k_w), shading='auto', cmap='turbo')
plt.colorbar(label='log10 Energy Density')
plt.xlabel('Wavenumber (cycle/m)')
plt.ylabel('Frequency (Hz)')
plt.title('Wavenumber-Frequency Spectrum')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
