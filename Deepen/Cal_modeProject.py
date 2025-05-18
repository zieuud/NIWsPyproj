import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


uv_ni = np.load(r'../MoorData/ADCP_uv_sd.npz')
u_ni = uv_ni['u_sd']
v_ni = uv_ni['v_sd']
ke_ni = 1 / 2 * 1025 * (u_ni ** 2 + v_ni ** 2)
moor = np.load(r'../MoorData/ADCP_uv.npz')
depth = moor['depth_adcp']
time = moor['mtime_adcp']
nz = len(depth)
nt = len(time)
dz = 8
dt = 3600

nmodes = 11
pmodes = np.load(r'../ReanaData/WOA23_pmodes_moorGrid_yearly.npz')['pmodes'][:, 4:]
pmodes_norm = np.zeros(np.shape(pmodes))
pmodes_norm[0, :] = pmodes[0, :]

for i in range(1, nmodes):
    xmaxabs = np.nanmax(np.abs(pmodes[i, :]))
    # pmodes_norm[i, :] = pmodes[i, :] / xmaxabs
    pmodes_norm[i, :] = pmodes[i, :]
    if pmodes_norm[i, 0] < 0:
        pmodes_norm[i, :] = -pmodes_norm[i, :]

fig = plt.figure(1, figsize=(10, 12))
gs = gridspec.GridSpec(2, 6, hspace=0, wspace=0)
for i in range(nmodes):
    if i >= 6:
        j = i + 1
    else:
        j = i
    ax = fig.add_subplot(gs[j])
    ax.plot(pmodes_norm[i, :].T, depth, 'k-')
    if i == 0 or i == 6:
        ax.set_ylabel('Depth (m)')
        ax.set_yticks([0, -500, -1000, -1500, -2000])
    # else:
        # ax.set_yticks([])
    ax.set_xticks([])
    ax.plot([0, 0], [0, -2000], 'k--')
    ax.text(0, -1800, 'mode {}'.format(i), fontsize=16, va='center', ha='center')
    # ax.set_ylim(-2000, 0)
plt.savefig(r'figures/Vertical Modes Structure (WOA23, 10 modes).png', dpi=300, bbox_inches='tight')

# projection
pmodes = np.tile(pmodes_norm, (nt, 1, 1))
u_mod = np.zeros((nt, nmodes, nz)) * np.nan
v_mod = np.zeros((nt, nmodes, nz)) * np.nan
ke_mod = np.zeros((nt, nmodes, nz)) * np.nan
for t in range(nt):
    valid_indices = np.where(~np.isnan(u_ni[t, :]))[0]
    u_mod_coeff = np.linalg.lstsq(pmodes[t, :, valid_indices], u_ni[t, valid_indices], rcond=None)[0]
    u_mod[t, :, valid_indices] = u_mod_coeff * pmodes[t, :, valid_indices]
    v_mod_coeff = np.linalg.lstsq(pmodes[t, :, valid_indices], v_ni[t, valid_indices], rcond=None)[0]
    v_mod[t, :, valid_indices] = v_mod_coeff * pmodes[t, :, valid_indices]
    ke_mod[t, :, :] = 1 / 2 * 1025 * (u_mod[t, :, :] ** 2 + v_mod[t, :, :] ** 2)

plt.figure(2, figsize=(20, 10))
for i in range(0, nmodes-1):
    if i == 0:
        plt.subplot(5, 2, i+1)
        plt.pcolormesh(time, depth, ke_ni.T, cmap='Oranges', vmin=0, vmax=10)
        plt.colorbar()
    else:
        plt.subplot(5, 2, i+1)
        c = plt.pcolormesh(time, depth, np.squeeze(ke_mod[:, i, :].T), cmap='Oranges', vmin=0, vmax=5)
        cb = plt.colorbar(c)
        cb.set_label(r'$KE_{NI}^{WKB}$ $(J/m^{3})$')
# plt.savefig(r'figures/Vertical Profile of Modes Kinetic Energy (WAO23, semi-diurnal).png', dpi=300, bbox_inches='tight')

plt.figure(3, figsize=(8, 6))
dz = depth[1] - depth[0]
KE_mod_dita = np.nanmean(np.nansum(ke_mod[:3960, :, :], -1) * 8, 0)
plt.figure(3)
plt.bar(range(0, nmodes), KE_mod_dita)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.xlabel('modes')
plt.ylabel('$KE_{ni}^{wkb}$ $(J·m^{-3})$')
# plt.savefig(r'figures/Depth-integrated Time-averaged Modes Kinetic Energy (WOA23, semi-diurnal).png', dpi=300, bbox_inches='tight')

# plt.figure(4, figsize=(8, 6))
# plt.plot(time[:3960], np.nansum(ke_mod[:3960, 1, :], -1))
# plt.plot(time[:3960], np.nansum(ke_mod[:3960, 2, :], -1))
# plt.plot(time[:3960], np.nansum(ke_mod[:3960, 3, :], -1))
# plt.plot(time[:3960], np.nansum(ke_mod[:3960, 4, :], -1))
# plt.legend(['1', '2', '3', '4'])

plt.show()
