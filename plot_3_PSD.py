import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import gsw


def to_complex(u, v):
    """将u和v分量转换为复数格式并处理NaN值"""
    uv = u + 1j * v
    return np.nan_to_num(uv)


def multitaper_spectral_analysis(uv, sample_rate=1.0, time_bandwidth=4, num_tapers=7):
    """
    Multitaper spectral analysis using DPSS windows
    Parameters:
        uv: Complex time series (time × depth)
        sample_rate: Sampling frequency (Hz)
        time_bandwidth: NW product (typical values 3-4)
        num_tapers: Number of tapers to use (usually 2*NW-1)

    Returns:
        pos_freq: Positive frequency array
        pos_spectrum: Positive rotary spectrum (counter-clockwise)
        neg_spectrum: Negative rotary spectrum (clockwise)
        cross_spectrum: Cross spectrum between components
    """
    n_time, n_depth = uv.shape
    nfft = n_time  # Can use zero-padding for better frequency resolution
    # Generate DPSS windows
    dpss_windows = signal.windows.dpss(n_time, NW=time_bandwidth, Kmax=num_tapers)
    # Get full frequency range (including negative frequencies)
    full_freqs = fftfreq(nfft, d=1.0 / sample_rate)
    # Initialize outputs (only positive frequencies)
    pos_freq = full_freqs[full_freqs >= 0]  # Positive frequencies
    pos_spectrum = np.zeros((n_depth, len(pos_freq)))
    neg_spectrum = np.zeros((n_depth, len(pos_freq)))
    cross_spectrum = np.zeros((n_depth, len(pos_freq)), dtype=complex)
    for i in range(n_depth):
        time_series = uv[:, i]
        psd_sum = np.zeros(nfft, dtype=complex)
        # Compute periodogram for each taper
        for window in dpss_windows:
            fft_vals = fft(window * time_series, n=nfft)
            psd_sum += fft_vals * np.conj(fft_vals)  # Power spectral density
        psd_avg = psd_sum / len(dpss_windows)  # Average across tapers
        # Separate positive and negative frequencies
        pos_psd = psd_avg[full_freqs >= 0]  # Positive frequencies (counter-clockwise)
        neg_psd = psd_avg[full_freqs < 0][::-1]  # Negative frequencies (clockwise), reversed

        pos_spectrum[i] = np.abs(pos_psd)  # Positive rotary spectrum
        neg_spectrum[i] = np.abs(neg_psd)  # Negative rotary spectrum
        cross_spectrum[i] = pos_psd * np.conj(neg_psd)  # Cross spectrum

    return pos_freq, pos_spectrum, neg_spectrum, cross_spectrum


adcp = np.load(r'MoorData/ADCP_uv.npz')
depth = adcp['depth_adcp']
time = adcp['mtime_adcp']
u = adcp['u']
v = adcp['v']
lat_moor = 36.23
lon_moor = -32.75
f = 2 * 7.292 * 1e-5 * np.sin(np.deg2rad(lat_moor)) / (2 * np.pi)
uv_complex = to_complex(u, v)
freq, spp, snn, spn = multitaper_spectral_analysis(uv_complex, 1. / 3600.)
spp_mean = np.nanmean(spp, 0)
snn_mean = np.nanmean(snn, 0)

plt.rcParams['font.size'] = 16
plt.figure(1, figsize=(15, 12))
plt.subplot(2, 1, 1)
plt.loglog(freq, spp_mean, label='CW')
plt.loglog(freq, snn_mean, label='CCW')
plt.axvspan(0.8 * f, 1.2 * f, color='gray', alpha=0.3)
plt.plot([f, f], [1e-4, 1e1], 'k-')
plt.text(f, 1e1, 'f', fontsize=16)
plt.plot([1 / 12.42 / 3600, 1 / 12.42 / 3600], [1e-4, 1e1], 'k--')
plt.text(1 / 12.42 / 3600., 1e1, '$M_{2}$', fontsize=16)
plt.xlim([freq.min(), freq.max()])
plt.ylim([1e-4, 1e1])
plt.xlabel('Frequency (Hz)', fontsize=16)
plt.ylabel('PSD (($m·s^{-1})^{2}Hz^{-1}$)', fontsize=16)
plt.text(3e-8, 7.5, 'a', ha='left', va='top',
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
plt.legend(fontsize=16)

plt.subplot(2, 3, 4)
plt.pcolormesh(freq, depth, spp, norm=LogNorm(vmin=1e-5, vmax=1e0), rasterized=True)
plt.xscale('log')
plt.xlim([1e-6, 1e-4])
plt.ylim([-2000, 0])
plt.colorbar(orientation='horizontal', location='bottom', label='CW PSD ($(m·s^{-1})^{2}Hz^{-1}$)')
plt.xlabel('Frequency (Hz)', fontsize=16)
plt.ylabel('Depth (m)', fontsize=16)
plt.plot([f, f], [0, -2000], 'k-')
plt.text(f, 1., 'f', fontsize=16)
plt.plot([1 / 12.42 / 3600., 1 / 12.42 / 3600.], [0, -2000], 'k--')
plt.text(1 / 12.42 / 3600., 1., '$M_{2}$', fontsize=16)
plt.text(1.1e-6, -100, 'b', ha='left', va='top',
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

plt.subplot(2, 3, 5)
plt.pcolormesh(freq, depth, snn, norm=LogNorm(vmin=1e-5, vmax=1e0), rasterized=True)
plt.xscale('log')
plt.xlim([1e-6, 1e-4])
plt.ylim([-2000, 0])
plt.colorbar(orientation='horizontal', location='bottom', label='CCW PSD ($(m·s^{-1})^{2}Hz^{-1}$)')
plt.xlabel('Frequency (Hz)', fontsize=16)
plt.plot([f, f], [0, -2000], 'k-')
plt.text(f, 1., 'f', fontsize=16)
plt.plot([1 / 12.42 / 3600., 1 / 12.42 / 3600.], [0, -2000], 'k--')
plt.text(1 / 12.42 / 3600., 1., '$M_{2}$', fontsize=16)
plt.text(1.1e-6, -100, 'c', ha='left', va='top',
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

plt.subplot(2, 3, 6)
plt.pcolormesh(freq, depth, snn / spp, norm=LogNorm(vmin=1e-2, vmax=1e2), cmap='coolwarm', rasterized=True)
plt.xscale('log')
plt.xlim([1e-6, 1e-4])
plt.ylim([-2000, 0])
plt.colorbar(orientation='horizontal', location='bottom', label='CW/CCW')
plt.xlabel('Frequency (Hz)', fontsize=16)
plt.plot([f, f], [0, -2000], 'k-')
plt.text(f, 1., 'f', fontsize=16)
plt.plot([1 / 12.42 / 3600., 1 / 12.42 / 3600.], [0, -2000], 'k--')
plt.text(1 / 12.42 / 3600., 1., '$M_{2}$', fontsize=16)
plt.text(1.1e-6, -100, 'd', ha='left', va='top',
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

plt.savefig('figures/fig_3_PSD.jpg', dpi=300, bbox_inches='tight')
plt.savefig(r'figuresFinal/fig_3_PSD.png', dpi=300, bbox_inches='tight')
plt.savefig(r'figuresFinal/fig_3_PSD.pdf', bbox_inches='tight')
plt.show()

print('c')
