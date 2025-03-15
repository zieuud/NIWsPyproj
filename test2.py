import numpy as np
import matplotlib.pyplot as plt
from spectrum import pmtm


adcp = np.load(r'ADCP_uv.npz')
u = adcp['u']
v = adcp['v']
u = np.nanmean(u, 0)
v = np.nanmean(v, 0)
depth = adcp['depth']
time = adcp['mtime']
t = time * 86400
dt = 3600
fs = 1 / dt

# 复数形式的流速
z = u + 1j * v
# 顺时针 (CW) 和 逆时针 (CCW) 分量
z_CW = 0.5 * (u + 1j * v)
z_CCW = 0.5 * (u - 1j * v)
# Multitaper 计算
Sk_CW, weights_CW, _ = pmtm(z_CW, NW=3, method='adapt', show=False)
Sk_CCW, weights_CCW, _ = pmtm(z_CCW, NW=3, method='adapt', show=False)
# 平均权重后的功率谱
psd_CW = np.mean(np.abs(Sk_CW.T * weights_CW), axis=1)
psd_CCW = np.mean(np.abs(Sk_CCW.T * weights_CCW), axis=1)
# 频率轴 (f = k / NFFT * fs)
freqs = np.linspace(0, fs/2, 8192)

# 绘图
plt.figure(figsize=(8, 6))

plt.plot(freqs, psd_CW, label='Clockwise (CW)', color='blue')
plt.plot(freqs, psd_CCW, label='Counter-Clockwise (CCW)', color='red')

# plt.xlim(0, 0.5)  # 通常关注科氏频率附近
plt.yscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('Rotary Power Spectrum (Multitaper)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
