import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from scipy import interpolate

vorticity_moor_hourly = np.load(r'ReanaData\GLORYS_vorticity_layers.npy')
strain_moor_hourly = np.load(r'ReanaData\GLORYS_strain_layers.npy')
moorData = np.load(r'ADCP_uv_ni.npz')
KE_ni = moorData['KE_ni']
adcp0 = np.load(r'ADCP_uv.npz')
moorDepth = adcp0['depth']
lat_moor = 36.23
fi = 2 * 7.292e-5 * np.sin(lat_moor/180*np.pi)
vf = vorticity_moor_hourly / fi
sf = strain_moor_hourly / fi
# ---------- plot jPDF ----------
# 展平
vorticity_flat = vf.flatten()
strain_flat = sf.flatten()
KE_flat = KE_ni.flatten()
# 分区间
x_bins = np.linspace(vorticity_flat.min(), vorticity_flat.max(), 50)  # vorticity的区间
y_bins = np.linspace(strain_flat.min(), strain_flat.max(), 50)  # strain的区间
# 计算每个bin的KE总和
hist, xedges, yedges = np.histogram2d(vorticity_flat, strain_flat, bins=[x_bins, y_bins], weights=KE_flat)
hist_density = hist / (np.nanmax(hist) - np.nanmin(hist))
# 绘制热图
plt.figure(figsize=(12, 6))
plt.imshow(hist_density.T, origin='lower', aspect='auto', cmap='Blues', extent=[xedges[0], -xedges[0], yedges[0], yedges[-1]],
           norm=LogNorm(1e-4, 1))
plt.plot(yedges, yedges, 'k--')
plt.plot(-yedges, yedges, 'k--')
# ---------- beautify ----------
plt.colorbar(label='PDF (%)')
plt.xlim(-0.2, 0.2)
plt.ylim(0, 0.2)
plt.xlabel(r'$\zeta/f$')
plt.ylabel(r'$\sigma/f$')
plt.title('jPDF')
# plt.savefig(r'figures\jPDF.jpg', dpi=350)
plt.show()
# ---------- plot depth distribution of different dominant regions ----------
AVD = np.zeros(245)
AVD_count = np.zeros(245)
SD = np.zeros(245)
SD_count = np.zeros(245)
CVD = np.zeros(245)
CVD_count = np.zeros(245)
for i in range(245):
    for j in range(6650):
        if np.isnan(KE_ni[j, i]):
            continue
        if vf[j, i] == 0:
            SD[i] += KE_ni[j, i]
            SD_count[i] += 1
        elif -1 <= sf[j, i]/vf[j, i] < 0:
            AVD[i] += KE_ni[j, i]
            AVD_count[i] += 1
        elif sf[j, i]/vf[j, i] < -1 or sf[j, i]/vf[j, i] > 1:
            SD[i] += KE_ni[j, i]
            SD_count[i] += 1
        else:
            CVD[i] += KE_ni[j, i]
            CVD_count[i] += 1
plt.figure(figsize=(8, 6))
annualAVD = AVD/AVD_count
x_valid = np.arange(len(annualAVD))[~np.isnan(annualAVD)]
y_valid = annualAVD[~np.isnan(annualAVD)]
interp_func = interpolate.interp1d(x_valid, y_valid, kind='linear', fill_value="extrapolate")
annualAVD_filled = annualAVD.copy()
annualAVD_filled[np.isnan(annualAVD)] = interp_func(np.arange(len(annualAVD))[np.isnan(annualAVD)])

annualSD = SD/SD_count
x_valid = np.arange(len(annualSD))[~np.isnan(annualSD)]
y_valid = annualSD[~np.isnan(annualSD)]
interp_func = interpolate.interp1d(x_valid, y_valid, kind='linear', fill_value="extrapolate")
annualSD_filled = annualSD.copy()
annualSD_filled[np.isnan(annualSD)] = interp_func(np.arange(len(annualSD))[np.isnan(annualSD)])

annualCVD = CVD/CVD_count
x_valid = np.arange(len(annualCVD))[~np.isnan(annualCVD)]
y_valid = annualCVD[~np.isnan(annualCVD)]
interp_func = interpolate.interp1d(x_valid, y_valid, kind='linear', fill_value="extrapolate")
annualCVD_filled = annualCVD.copy()
annualCVD_filled[np.isnan(annualCVD)] = interp_func(np.arange(len(annualCVD))[np.isnan(annualCVD)])

plt.plot(annualAVD_filled, moorDepth)
plt.plot(annualSD_filled, moorDepth)
plt.plot(annualCVD_filled, moorDepth)
plt.legend(['AVD', 'SD', 'CVD'])
plt.show()
# for debugging
print('c')
