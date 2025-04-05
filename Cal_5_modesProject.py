import numpy as np
from func_1_dynmodes import dynmodes

uv_ni = np.load(r'MoorData/ADCP_uv_ni_wkb_byFusionNsq.npz')
u_ni_wkb = uv_ni['u_ni_wkb']
v_ni_wkb = uv_ni['v_ni_wkb']
moor = np.load(r'MoorData/ADCP_uv.npz')
depth = moor['depth_adcp']
time = moor['mtime_adcp']
nz = len(depth)
nt = len(time)
Nsq = np.load(r'ReanaData/WOA23_stratification_adcpGrid.npz')['Nsq_fusion_adcpGrid']
nmodes = 11

u_mod = np.zeros((nt, nmodes, nz))
v_mod = np.zeros((nt, nmodes, nz))
ke_mod = np.zeros((nt, nmodes, nz))
for t in range(nt):
    wmodes, pmodes, ce = dynmodes(np.squeeze(Nsq[t, :]), depth[:], nmodes)
    u_mod_coeff = np.linalg.lstsq(pmodes.T, np.squeeze(u_ni_wkb[t, :]))[0]
    u_mod[t, :, :] = np.dot(u_ni_wkb[t, :].reshape(-1, 1), u_mod_coeff.reshape(1, -1)).T
    v_mod_coeff = np.linalg.lstsq(pmodes.T, np.squeeze(v_ni_wkb[t, :]))[0]
    v_mod[t, :, :] = np.dot(v_ni_wkb[t, :].reshape(-1, 1), v_mod_coeff.reshape(1, -1)).T
    ke_mod[t, :, :] = 1 / 2 * 1025 * (u_mod[t, :, :] ** 2 + v_mod[t, :, :] ** 2)
# np.savez(r'ADCP_uv_10modes.npz', u_mod=u_mod, v_mod=v_mod, KE_mod=ke_mod)
print('c')