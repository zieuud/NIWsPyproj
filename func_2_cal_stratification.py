import numpy as np
import gsw


def cal_stratification(nt, nz, z, s, t, lat_moor, lon_moor):
    ct = np.zeros((nt, nz))
    sig0 = np.zeros((nt, nz))
    N2 = np.zeros((nt, nz - 1))
    dtdz = np.zeros((nt, nz - 1))
    for m in range(nt):
        p = gsw.p_from_z(z, lat_moor)
        SA = gsw.SA_from_SP(s[m, :], p, lon_moor, lat_moor)
        CT = gsw.CT_from_t(SA, t[m, :], p)
        [N2[m, :], p_mid] = gsw.Nsquared(SA, CT, p, lat_moor)
        ct[m, :] = CT
        ze = gsw.z_from_p(p_mid.data, lat_moor)
        sig0[m, :] = gsw.sigma0(SA, CT)
        dtdz[m, :] = np.diff(CT) / np.diff(z)
    N2[N2 < 0] = 1e-8
    return ct, sig0, N2, dtdz, ze