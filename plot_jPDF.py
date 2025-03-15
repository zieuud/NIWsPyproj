import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from scipy import interpolate

# use GLORYS data
# vorticity_moor_hourly = np.load(r'ReanaData\GLORYS_vorticity.npy')[:, :180]
# strain_moor_hourly = np.load(r'ReanaData\GLORYS_strain.npy')[:, :180]
# use AVISO data
vorticity_moor_hourly = np.load(r'ReanaData\AVISO_vorticity1.npy')
vorticity_moor_hourly = np.tile(vorticity_moor_hourly, (180, 1)).T
strain_moor_hourly = np.load(r'ReanaData\AVISO_strain1.npy')
strain_moor_hourly = np.tile(strain_moor_hourly, (180, 1)).T

moorData = np.load(r'ADCP_uv_ni_wkb.npz')
KE_ni = moorData['KE_ni_wkb'][:, :180]
adcp0 = np.load(r'ADCP_uv.npz')
moorDepth = adcp0['depth']
lat_moor = 36.23
fi = 2 * 7.292e-5 * np.sin(lat_moor/180*np.pi)
vf = vorticity_moor_hourly / fi
sf = strain_moor_hourly / fi
# ---------- plot jPDF ----------
vorticity_flat = vf.flatten()
strain_flat = sf.flatten()
KE_flat = KE_ni.flatten()
indices = np.isnan(KE_flat)
KE_flat = KE_flat[~indices]
vorticity_flat = vorticity_flat[~indices]
strain_flat = strain_flat[~indices]

x_bins = np.linspace(vorticity_flat.min(), vorticity_flat.max(), 50)
y_bins = np.linspace(strain_flat.min(), strain_flat.max(), 50)

hist, xedges, yedges = np.histogram2d(vorticity_flat, strain_flat, bins=[x_bins, y_bins], weights=KE_flat)
hist_density = hist / (np.nanmax(hist) - np.nanmin(hist))
indices1 = (hist_density == 0)
indices2 = np.isnan(hist_density)
# plot
plt.figure(figsize=(12, 6))
# ----- probability1 -----
plt.imshow(hist_density.T, origin='lower', aspect='auto', cmap='Blues', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], norm=LogNorm(1e-4, 1))
# ----- probability2 -----
# X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
# levels = np.logspace(-5, 1, 6)
# plt.contourf(X, Y, hist_density.T, cmap="Blues", levels=levels, norm=LogNorm())
# ----- NIKE absolute value -----
hist[hist == 0] = np.nan
# plt.imshow(hist.T, origin='lower', aspect='auto', cmap='Blues', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin=0, vmax=10e3)
plt.plot([0, 0.3], [0, 0.3], 'k--')
plt.plot([-0.3, 0], [0.3, 0], 'k--')
# ---------- beautify ----------
plt.colorbar(label='PDF (%)')
plt.xlim(-0.3, 0.3)
plt.ylim(0, 0.3)
plt.xlabel(r'$\zeta/f$')
plt.ylabel(r'$\sigma/f$')
plt.title('jPDF')
# plt.savefig(r'figures\jPDF_NIKE.jpg', dpi=300)
plt.show()
# ---------- plot depth distribution of different dominant regions ----------
nz = len(moorDepth[:180])
AVD = np.zeros(nz)
AVD_count = np.zeros(nz)
SD = np.zeros(nz)
SD_count = np.zeros(nz)
CVD = np.zeros(nz)
CVD_count = np.zeros(nz)
for i in range(nz):
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
plt.figure(figsize=(6, 8))
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

plt.plot(annualAVD_filled, moorDepth[:180])
plt.plot(annualSD_filled, moorDepth[:180])
plt.plot(annualCVD_filled, moorDepth[:180])
plt.legend(['AVD', 'SD', 'CVD'])
plt.savefig(r'figures\depth distribution of different dominant regions.jpg', dpi=300)
plt.show()
# for debugging
print('c')
