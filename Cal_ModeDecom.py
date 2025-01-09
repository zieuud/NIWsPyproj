import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import gsw
import utils


caseName = ['2_3', '2_4', '3_3', '3_4', '4_3', '4_4']
# caseName = ['2']
coord = loadmat(r'L:\NIWs\ExtractData\CaseV5\ParallelLine4_coord.mat')
z_full = coord['z_line'][:, ::-1, :]
for case in caseName:
    if case == '2':
        uvr = loadmat(r'L:\NIWs\ExtractData\CaseV5\ParallelLine4_uvr.mat')
        uv_filt = loadmat(r'L:\NIWs\ExtractData\CaseV5\ParallelLine4_uv_filter.mat')
    else:
        uvr = loadmat(r'L:\NIWs\ExtractData\CaseV5_{}\ParallelLine4_uvr.mat'.format(case))
        uv_filt = loadmat(r'L:\NIWs\ExtractData\CaseV5_{}\ParallelLine4_uv_filter.mat'.format(case))
    u = uv_filt['u_pline_filter'][:]
    v = uv_filt['v_pline_filter'][:]
    rho_full = uvr['rho_pline'][:, ::-1, :] + 1000
    # lat = coord['lat_line'][:, :]
    # f = gsw.f(lat)
    # omega = 0.8*f
    [wmodes, ce, e] = utils.mode_decomposition_3d(rho_full, z_full, 3)
    np.savez(r'OtherProj\mode_decom_{}.npz'.format(case), wmodes=wmodes, ce=ce, e=e)

    u_pline_filter_mod = utils.u_decom(u, wmodes)
    v_pline_filter_mod = utils.u_decom(v, wmodes)
    np.savez(r'OtherProj\ParallelLine4_uv_filter_mod_{}.npz'.format(case), u_pline_filter_mod=u_pline_filter_mod,
             v_pline_filter_mod=v_pline_filter_mod)


print('test')
