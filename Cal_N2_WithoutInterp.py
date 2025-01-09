import numpy as np
import gsw
from datetime import datetime, timedelta
import scipy.interpolate as itp
import matplotlib.pyplot as plt

woa23 = np.load(r'ReanaData\WOA23_st.npz')
s = woa23['s']
t = woa23['t']
z = -woa23['z']
[lat, lon] = woa23['loc']
sig0 = np.zeros((12, 57))
N2 = np.zeros((12, 56))
for m in range(1, 13):
    p = gsw.p_from_z(z, lat)
    SA = gsw.SA_from_SP(s[m - 1, :], p, lon, lat)
    CT = gsw.CT_from_t(SA, t[m - 1, :], p)
    [N2[m - 1, :], p_mid] = gsw.Nsquared(SA, CT, p, lat)
    ze = gsw.z_from_p(p_mid.data, lat)
    sig0[m - 1, :] = gsw.sigma0(SA, CT)

print('cut')
