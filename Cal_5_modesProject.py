import matplotlib.pyplot as plt
import numpy as np
from func_1_dynmodes import dynmodes

maxIdx = 181
uv_ni = np.load(r'MoorData/ADCP_uv_ni_wkb.npz')
u_ni = uv_ni['u_ni_wkb'][:, :maxIdx]
v_ni = uv_ni['v_ni_wkb'][:, :maxIdx]
moor = np.load(r'MoorData/ADCP_uv.npz')
depth = moor['depth_adcp'][:maxIdx]
time = moor['mtime_adcp']
nz = len(depth)
nt = len(time)

nmodes = 11
pmodes = np.load(r'ReanaData/WOA23_pmodes_moorGrid.npz')['pmodes']

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

np.savez(r'MoorData/ADCP_uv_ni_10bcmodes.npz', u_mod=u_mod, v_mod=v_mod, ke_mod=ke_mod)
print('c')
