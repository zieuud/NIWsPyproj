import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

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

# u_mod = np.zeros((nt, nmodes, nz)) * np.nan
# v_mod = np.zeros((nt, nmodes, nz)) * np.nan
# ke_mod = np.zeros((nt, nmodes, nz)) * np.nan
#
# for t in range(nt):
#     valid_indices = np.where(~np.isnan(u_ni[t, :]))[0]
#     u_mod_coeff = np.linalg.lstsq(pmodes[t, :, valid_indices], u_ni[t, valid_indices], rcond=None)[0]
#     u_mod[t, :, valid_indices] = u_mod_coeff * pmodes[t, :, valid_indices]
#     v_mod_coeff = np.linalg.lstsq(pmodes[t, :, valid_indices], v_ni[t, valid_indices], rcond=None)[0]
#     v_mod[t, :, valid_indices] = v_mod_coeff * pmodes[t, :, valid_indices]
#     ke_mod[t, :, :] = 1 / 2 * 1025 * (u_mod[t, :, :] ** 2 + v_mod[t, :, :] ** 2)

# np.savez(r'MoorData/ADCP_uv_ni_10bcmodes.npz', u_mod=u_mod, v_mod=v_mod, ke_mod=ke_mod)

# ---------- energy flux projection u v p ----------
# fluxParams = np.load(r'MoorData/EnergyFlux.npz')
# up = fluxParams['up']
# vp = fluxParams['vp']
# pp = fluxParams['pp']
#
# depthFlux = depth[9:]
# nzFlux = len(depthFlux)
# pmodesFlux = pmodes[:, :, 9:maxIdx]
#
# up_mod = np.zeros((nt, nmodes, nzFlux)) * np.nan
# vp_mod = np.zeros((nt, nmodes, nzFlux)) * np.nan
# pp_mod = np.zeros((nt, nmodes, nzFlux)) * np.nan
#
# for t in range(nt):
#     valid_indices = np.where(~np.isnan(up[t, :]))[0]
#     up_mod_coeff = np.linalg.lstsq(pmodesFlux[t, :, valid_indices], up[t, valid_indices], rcond=None)[0]
#     up_mod[t, :, valid_indices] = up_mod_coeff * pmodesFlux[t, :, valid_indices]
#     vp_mod_coeff = np.linalg.lstsq(pmodesFlux[t, :, valid_indices], vp[t, valid_indices], rcond=None)[0]
#     vp_mod[t, :, valid_indices] = vp_mod_coeff * pmodesFlux[t, :, valid_indices]
#     pp_mod_coeff = np.linalg.lstsq(pmodesFlux[t, :, valid_indices], pp[t, valid_indices], rcond=None)[0]
#     pp_mod[t, :, valid_indices] = pp_mod_coeff * pmodesFlux[t, :, valid_indices]

# for t in range(nt):
#     valid_indices = np.where(~np.isnan(up[t, :]))[0]
#     ridge = Ridge(alpha=0.1)
#     ridge.fit(pmodesFlux[t, :, valid_indices], up[t, valid_indices])
#     up_mod_coeff = ridge.coef_
#     up_mod[t, :, valid_indices] = up_mod_coeff * pmodesFlux[t, :, valid_indices]
#     ridge.fit(pmodesFlux[t, :, valid_indices], vp[t, valid_indices])
#     vp_mod_coeff = ridge.coef_
#     vp_mod[t, :, valid_indices] = vp_mod_coeff * pmodesFlux[t, :, valid_indices]
#     ridge.fit(pmodesFlux[t, :, valid_indices], pp[t, valid_indices])
#     pp_mod_coeff = ridge.coef_
#     pp_mod[t, :, valid_indices] = pp_mod_coeff * pmodesFlux[t, :, valid_indices]

# np.savez(r'MoorData/EnergyFlux_10bcmodes.npz', up_mod=up_mod, vp_mod=vp_mod, pp_mod=pp_mod)

# ---------- energy flux projection fx fy ----------
fh = np.load(r'MoorData/EnergyFlux.npz')
fx = fh['fx']
fy = fh['fy']

depthFlux = depth[9:]
nzFlux = len(depthFlux)
pmodesFlux = pmodes[:, :, 9:maxIdx]

fx_mod = np.zeros((nt, nmodes, nz))
fy_mod = np.zeros((nt, nmodes, nz))

# The least squares regression
# for t in range(nt):
#     valid_indices = np.where(~np.isnan(fx[t, :]))[0]
#     fx_mod_coeff = np.linalg.lstsq(pmodesFlux[t, :, valid_indices], fx[t, valid_indices], rcond=None)[0]
#     fx_mod[t, :, valid_indices] = fx_mod_coeff * pmodesFlux[t, :, valid_indices]
#     fy_mod_coeff = np.linalg.lstsq(pmodesFlux[t, :, valid_indices], fy[t, valid_indices], rcond=None)[0]
#     fy_mod[t, :, valid_indices] = fy_mod_coeff * pmodesFlux[t, :, valid_indices]
# Ridge regression
for t in range(nt):
    valid_indices = np.where(~np.isnan(fx[t, :]))[0]
    ridge1 = Ridge(alpha=1e-3)
    ridge1.fit(pmodesFlux[t, :, valid_indices], fx[t, valid_indices])
    fx_mod_coeff = ridge1.coef_
    fx_mod[t, :, valid_indices] = fx_mod_coeff * pmodesFlux[t, :, valid_indices]
    ridge2 = Ridge(alpha=1e-2)
    ridge2.fit(pmodesFlux[t, :, valid_indices], fy[t, valid_indices])
    fy_mod_coeff = ridge2.coef_
    fy_mod[t, :, valid_indices] = fy_mod_coeff * pmodesFlux[t, :, valid_indices]

np.savez(r'MoorData/EnergyFlux_10bcmodes_fhProj_Ridge.npz', fx_mod=fx_mod, fy_mod=fy_mod)

print('c')
