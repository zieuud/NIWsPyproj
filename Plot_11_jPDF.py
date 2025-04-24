import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from scipy import interpolate


def smooth(profile):
    x_valid = np.arange(len(profile))[~np.isnan(profile)]
    y_valid = profile[~np.isnan(profile)]
    interp_func = interpolate.interp1d(x_valid, y_valid, kind='linear', fill_value="extrapolate")
    profile_filled = profile.copy()
    profile_filled[np.isnan(profile)] = interp_func(np.arange(len(profile))[np.isnan(profile)])
    return profile_filled


# use GLORYS data
# vorticity_moor_hourly = np.load(r'ReanaData\GLORYS_vorticity.npy')
# strain_moor_hourly = np.load(r'ReanaData\GLORYS_strain.npy')
# use AVISO data
vorticity_moor_hourly = np.load(r'ReanaData\AVISO_vorticity1.npy')
# vorticity_moor_hourly = np.tile(vorticity_moor_hourly, (245, 1)).T
strain_moor_hourly = np.load(r'ReanaData\AVISO_strain1.npy')
# strain_moor_hourly = np.tile(strain_moor_hourly, (245, 1)).T

moorData = np.load(r'MoorData/ADCP_uv_ni_wkb.npz')
KE_ni = moorData['KE_ni_wkb']
KE_ni_dinteg = np.nansum(KE_ni * 8, 1)
adcp0 = np.load(r'MoorData/ADCP_uv.npz')
moorDepth = adcp0['depth_adcp']
lat_moor = 36.23
fi = 2 * 7.292e-5 * np.sin(lat_moor/180*np.pi)
vf = vorticity_moor_hourly / fi
sf = strain_moor_hourly / fi
# ---------- plot jPDF ----------
vorticity_flat = vf.flatten()
strain_flat = sf.flatten()
KE_flat = KE_ni_dinteg.flatten()
indices = np.isnan(KE_flat)
KE_flat = KE_flat[~indices]
vorticity_flat = vorticity_flat[~indices]
strain_flat = strain_flat[~indices]

x_bins = np.linspace(vorticity_flat.min(), vorticity_flat.max(), 30)
y_bins = np.linspace(strain_flat.min(), strain_flat.max(), 30)

hist, xedges, yedges = np.histogram2d(vorticity_flat, strain_flat, bins=[x_bins, y_bins], weights=KE_flat)
hist_density = hist / (np.nanmax(hist) - np.nanmin(hist)) * 100
# indices1 = (hist_density == 0)
# indices2 = np.isnan(hist_density)
# hist_density[indices1] = 1e-4
# plot
plt.figure(figsize=(12, 6))
# ----- probability1 -----
plt.imshow(hist_density.T, origin='lower', aspect='auto', cmap='Blues',
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], norm=LogNorm())
# ----- probability2 -----
# hist[hist == 0] = 1e-3
# X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
# levels = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]
# plt.contourf(X, Y, hist_density.T, cmap="Blues", levels=levels, norm=LogNorm())
# ----- NIKE absolute value -----
# hist[hist == 0] = np.nan
# plt.imshow(hist.T, origin='lower', aspect='auto', cmap='Blues',
#            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin=0, vmax=10e3)

plt.plot([0, 0.3], [0, 0.3], 'k--')
plt.plot([-0.3, 0], [0.3, 0], 'k--')
# ---------- beautify ----------
plt.colorbar(label='PDF (%)')
plt.xlim(-0.2, 0.2)
plt.ylim(0, 0.2)
plt.xlabel(r'$\zeta/f$')
plt.ylabel(r'$\sigma/f$')
plt.title('jPDF')
# plt.savefig(r'figures\fig_10_jPDF.jpg', dpi=300, bbox_inches='tight')
plt.show()
# ---------- plot depth distribution of different dominant regions ----------
nz = len(moorDepth[:180])
nt = 6650
AVD = np.zeros(nz)
AVD_count = np.zeros(nz) + 1
SD = np.zeros(nz)
SD_count = np.zeros(nz) + 1
CVD = np.zeros(nz)
CVD_count = np.zeros(nz) + 1
for i in range(nz):
    for j in range(nt):
        if np.isnan(KE_ni[j, i]):
            continue
        elif vf[j, i] == 0:
            SD[i] += KE_ni[j, i]
            SD_count[i] += 1
        elif vf[j, i] > 0:
            if vf[j, i] / sf[j, i] >= 1:
                CVD[i] += KE_ni[j, i]
                CVD_count[i] += 1
            else:
                SD[i] += KE_ni[j, i]
                SD_count[i] += 1
        else:
            if abs(vf[j, i]) / sf[j, i] >= 1:
                AVD[i] += KE_ni[j, i]
                AVD_count[i] += 1
            else:
                SD[i] += KE_ni[j, i]
                SD_count[i] += 1
AVD_filled = smooth(AVD)
annualAVD_filled = smooth(AVD / AVD_count)
SD_filled = smooth(SD)
annualSD_filled = smooth(SD / SD_count)
CVD_filled = smooth(CVD)
annualCVD_filled = smooth(CVD / CVD_count)

plt.figure(1, figsize=(6, 8))
plt.plot(annualAVD_filled, moorDepth[:180])
plt.plot(annualSD_filled, moorDepth[:180])
plt.plot(annualCVD_filled, moorDepth[:180])
plt.legend(['AVD', 'SD', 'CVD'])
plt.xlabel('time-averaged $KE_{ni}^{wkb}$ $(J·m^{-3})$')
plt.ylabel('depth (m)')
plt.savefig(r'figures\time-averaged KE profile of different dominant regions.jpg', dpi=300)
plt.show()

plt.figure(2, figsize=(6, 8))
plt.plot(AVD_filled, moorDepth[:180])
plt.plot(SD_filled, moorDepth[:180])
plt.plot(CVD_filled, moorDepth[:180])
plt.legend(['AVD', 'SD', 'CVD'])
plt.xlabel('total $KE_{ni}^{wkb}$ $(J·m^{-3})$')
plt.ylabel('depth (m)')
plt.savefig(r'figures\total KE profile of different dominant regions.jpg', dpi=300)
plt.show()
# for debugging
print('c')
