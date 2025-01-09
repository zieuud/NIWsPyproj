from datetime import datetime, timedelta
import netCDF4 as nc
import numpy as np
import scipy.interpolate as itp

aviso = nc.Dataset(r'K:\grad_proj\AVISO\cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D_1730875116530.nc')  # daily
lat = aviso.variables['latitude'][:]
lon = aviso.variables['longitude'][:]
time = aviso.variables['time'][:]
# sla = aviso.variables['sla']
u_geo = aviso.variables['ugos'][:]
v_geo = aviso.variables['vgos'][:]  # (time, lat, lon)
lat_moor = 36.23
lon_moor = -32.75
moorData = np.load('ADCP_uv.npz')
moorDate = moorData['mtime']
# print(np.shape(lon), np.shape(lat), np.shape(u_geo), np.shape(v_geo), np.shape(time))
# print(aviso.variables)
# ----------calculate the vorticity----------
dx = 110000./4.
dy = 110000./4.  # 单位m, 分辨率0.25度
latIdx = np.argwhere(lat == 36.125)
lonIdx = np.argwhere(lon == -32.875)
# ----------with simple interpolate----------
dvdx1 = (v_geo[:, latIdx-1, lonIdx] - v_geo[:, latIdx, lonIdx-1]) / dx
dvdx2 = (v_geo[:, latIdx, lonIdx] - v_geo[:, latIdx, lonIdx-1]) / dx
dudy1 = (u_geo[:, latIdx-1, lonIdx-1] - u_geo[:, latIdx, lonIdx-1]) / dy
dudy2 = (u_geo[:, latIdx-1, lonIdx] - u_geo[:, latIdx, lonIdx]) / dy
dvdx = (dvdx1 + dvdx2) / 2
dudy = (dudy1 + dudy2) / 2
vorticity_moor = np.squeeze(dvdx - dudy)
# ----------without interpolate----------
# dvdx = (v_geo[:, latIdx, lonIdx+1] - v_geo[:, latIdx, lonIdx-1]) / (2 * dx)
# dudy = (u_geo[:, latIdx+1, lonIdx] - u_geo[:, latIdx-1, lonIdx]) / (2 * dy)
# vorticity_moor = np.squeeze(dvdx - dudy)
# ----------with interpolate----------
# vorticity_moor_hourly = np.repeat(np.squeeze(vorticity_moor[21:366-31-30-6]), 24)[16:6672-6]
# vorticity_moor_hourly = np.array(vorticity_moor_hourly)
# dvdx = (v_geo[:, :, 2:] - v_geo[:, :, :-2]) / (2 * dx)
# dudy = (u_geo[:, 2:, :] - v_geo[:, :-2, :]) / (2 * dy)
# vorticity = dvdx[:, 1:-1, :] - dudy[:, :, 1:-1]

# ---------interpolate on time----------
# create aviso time
dates = [datetime(2016, 8, 1) + timedelta(days=i) for i in
         range((datetime(2017, 7, 31) - datetime(2016, 7, 31)).days + 1)]
avisoDate = [date.toordinal() for date in dates]
# print('test')
# # interpolate to moor location
# vorticity_moor = np.zeros(len(time))
# for t in range(len(time)):
#     itp_t = itp.RectBivariateSpline(lat[1:-1], lon[1:-1], vorticity[t, :, :])
#     vorticity_moor[t] = itp_t(lat_moor, lon_moor)
# # interpolate on time
itp_t = itp.interp1d(avisoDate, vorticity_moor)
vorticity_moor_hourly = itp_t(moorDate)

np.save(r'ReanaData\AVISO_vorticity5.npy', vorticity_moor_hourly)

# print(vorticity_moor)
