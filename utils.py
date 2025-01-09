import numpy as np


def mode_decomposition(rho_full, z_full, nmodes):
    nt = rho_full.shape[1]
    nz = rho_full.shape[0]
    # --- calculate the N2
    rho0 = 1025
    g = 9.81
    drho_dz = np.zeros((15, 600))
    for i in range(nt):
        drho_dz[:, i] = np.gradient(rho_full[:, i], z_full[:, i])
        drho_dz[drho_dz >= 0] = -1e-6
    N2_full = - (g / rho0) * drho_dz
    # N2_full[N2_full == 0] = 1e-10
    # --- mode decomposition
    e_full = np.zeros((nmodes, nt))
    ce_full = np.zeros((nmodes, nt))
    wmodes_full = np.zeros((nz, nmodes, nt))
    for t in range(nt):
        z = z_full[:, t]
        N2 = N2_full[:, t]
        # --- check surface value
        # if z[0] > 0:
        #     z[0] = 0
            # z = np.concatenate(([0], z))
            # N2 = np.concatenate(([N2[0]], N2))
        # --- calculate depth and spacing
        dz = z[:-1] - z[1:]  # spacing
        zm = z[:-1] - 0.5 * dz  # mid-point depth
        dzm = np.zeros(nz)  # mid-point spacing
        dzm[1:-1] = zm[:nz - 2] - zm[1:nz - 1]
        dzm[0] = dzm[1]
        dzm[-1] = dzm[-2]
        # --- set matrices
        A = np.zeros((nz, nz))
        B = np.zeros((nz, nz))
        for i in np.arange(1, nz - 1):
            A[i, i] = 1. / (dz[i - 1] * dzm[i]) + 1. / (dz[i] * dzm[i])
            A[i, i - 1] = -1. / (dz[i - 1] * dzm[i])
            A[i, i + 1] = -1. / (dz[i] * dzm[i])
        for i in np.arange(nz):
            B[i, i] = N2[i]
        A[0, 0] = -1.
        A[nz - 1, 0] = -1.
        # --- extract eigen stuff (w-modes)
        try:
            e, wmodes = np.linalg.eig(np.dot(np.linalg.inv(B), A))
        except:
            print(N2)
            print(z)
            print(rho_full[:, t])
        # --- only Real eigenvalues (physical meaning)
        ind = np.argwhere(np.imag(e) == 0)
        e = np.squeeze(e[ind])
        wmodes = np.squeeze(wmodes[:, ind])
        # --- no zeros eigenvalues
        ind = np.argwhere(e > 1e-10)
        e = np.squeeze(e[ind])
        wmodes = np.squeeze(wmodes[:, ind])
        # --- sort eigenvalues and corresponding eigenvectors
        ind = np.argsort(e)
        e = np.squeeze(e[ind])
        wmodes = np.squeeze(wmodes[:, ind])
        # --- calculate the eigen velocity
        ce = 1. / np.sqrt(e)
        # --- adds the barotropic mode

        bt_mode = np.ones((nz, 1))
        wmodes_full[:, :, t] = np.concatenate((bt_mode, wmodes), axis=1)[:, :nmodes]
        ce_full[:, t] = np.concatenate(([np.nan], ce))[:nmodes]
        e_full[:, t] = np.concatenate(([np.nan], e))[:nmodes]
    return wmodes_full, ce_full, e_full


def mode_decomposition_3d(rho_full, z_full, nmodes):
    nl = rho_full.shape[0]
    nz = rho_full.shape[1]
    nt = rho_full.shape[2]
    wmodes = np.zeros((nl, nz, nmodes, nt))
    ce = np.zeros((nl, nmodes, nt))
    e = np.zeros((nl, nmodes, nt))
    for l in range(nl):
        rho = np.squeeze(rho_full[l, :, :])
        z = np.squeeze(z_full[l, :, :])
        [wmodes[l, :, :, :], ce[l, :, :], e[l, :, :]] = mode_decomposition(rho, z, nmodes)
    return wmodes, ce, e


def u_decom(u, wmodes):
    [nl, nz, nmodes, nt] = wmodes.shape
    u_mod = np.zeros((nl, nz, nmodes, nt))
    for l in range(nl):
        for t in range(nt):
            wmodes_temp = wmodes[l, :, :, t]
            u_temp = u[l, :, t]
            u_cmod = np.linalg.lstsq(np.squeeze(wmodes_temp), np.squeeze(u_temp))[0]
            u_mod[l, :, :, t] = u_cmod*wmodes_temp
    return u_mod