import matplotlib.pyplot as plt
import numpy as np
from func_1_dynmodes import dynmodes

uv_ni = np.load(r'MoorData/ADCP_uv_ni_wkb.npz')
u_ni = uv_ni['u_ni_wkb']
v_ni = uv_ni['v_ni_wkb']
moor = np.load(r'MoorData/ADCP_uv.npz')
depth = moor['depth_adcp']
time = moor['mtime_adcp']
nz = len(depth)
nt = len(time)

nmodes = 11
pmodes = np.load(r'ReanaData/WOA23_modes_adcpGrid.npz')['pmodes']
pmodes[:, 0, :] = 1.

# pmodes = np.load(r'ReanaData/WOA23_pmodes.npy')
# pmodes = np.tile(pmodes, (6650, 1, 1))
# pmodes = np.load(r'ReanaData/WOA23_')
u_mod = np.zeros((nt, nmodes, nz)) * np.nan
v_mod = np.zeros((nt, nmodes, nz)) * np.nan
ke_mod = np.zeros((nt, nmodes, nz)) * np.nan
# for i in range(nmodes):
#     pmodes[:, i, :] /= np.linalg.norm(pmodes[:, i, :], axis=1, keepdims=True)
pmodes[pmodes[:, :, 0] < 0] = -pmodes[pmodes[:, :, 0] < 0]
for t in range(nt):
    valid_indices = np.where(~np.isnan(u_ni[t, :]))[0]
    u_mod_coeff = np.linalg.lstsq(pmodes[t, :, valid_indices], u_ni[t, valid_indices], rcond=None)[0]
    u_mod[t, :, valid_indices] = u_mod_coeff * pmodes[t, :, valid_indices]
    v_mod_coeff = np.linalg.lstsq(pmodes[t, :, valid_indices], v_ni[t, valid_indices], rcond=None)[0]
    v_mod[t, :, valid_indices] = v_mod_coeff * pmodes[t, :, valid_indices]
    ke_mod[t, :, :] = 1 / 2 * 1025 * (u_mod[t, :, :] ** 2 + v_mod[t, :, :] ** 2)

np.savez(r'MoorData/ADCP_uv_ni_10bcmodes.npz', u_mod=u_mod, v_mod=v_mod, ke_mod=ke_mod)
# np.savez(r'ADCP_uv_10modes.npz', u_mod=u_mod, v_mod=v_mod, KE_mod=ke_mod)
print('c')