import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt


aviso = nc.Dataset(r'K:\grad_proj\AVISO\cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D_1730875116530.nc')
lat = aviso.variables['latitude'][:]
lon = aviso.variables['longitude'][:]
time = aviso.variables['time'][:]
u_geo = aviso.variables['ugos'][:]
v_geo = aviso.variables['vgos'][:]
lat_moor = 36.23
lon_moor = -32.75
moorData = np.load('ADCP_uv.npz')
moorDate = moorData['mtime']
# ---------- calculate the vorticity and divergence of 4 points ----------
dx = 110000./4.
dy = 110000./4.
selections = [[-32.625, 36.375], [-32.875, 36.375], [-32.625, 36.125], [-32.875, 36.125]]
vorticitySelections = {}
divergenceSelections = {}
for locs in selections:
    lonIdx = np.argwhere(lon == locs[0])
    latIdx = np.argwhere(lat == locs[1])
    dvdx = (v_geo[:, latIdx, lonIdx + 1] - v_geo[:, latIdx, lonIdx - 1]) / (2 * dx)
    dudy = (u_geo[:, latIdx + 1, lonIdx] - u_geo[:, latIdx - 1, lonIdx]) / (2 * dy)
    vorticitySelections[tuple(locs)] = np.squeeze(dvdx - dudy)
    dudx = (u_geo[:, latIdx + 1, lonIdx] - u_geo[:, latIdx - 1, lonIdx]) / (2 * dx)
    dvdy = (v_geo[:, latIdx, lonIdx + 1] - v_geo[:, latIdx, lonIdx - 1]) / (2 * dy)
    divergenceSelections[tuple(locs)] = np.squeeze(dudx + dvdy)
# ---------- plot for checking ----------
plt.figure(1)
for vorticity in vorticitySelections.values():
    plt.plot(vorticity)
plt.title('vorticity (dv/dx - du/dy)')
plt.legend(['1', '2', '3', '4'])
plt.savefig(r'figures\check_vorticity.jpg', dpi=300)
plt.figure(2)
for divergence in divergenceSelections.values():
    plt.plot(divergence)
plt.title('divergence (du/dx + dv/dy)')
plt.legend(['1', '2', '3', '4'])
plt.savefig(r'figures\check_divergence.jpg', dpi=300)
plt.show()

print('c')


