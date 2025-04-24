import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean


fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], hspace=0.1, wspace=0.05)

ax1 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
topo = nc.Dataset(r'L:/graduation proj/data/GEBCO/GEBCO_2024.nc')
lon_topo = topo.variables['lon'][24000:43215:15]
lat_topo = topo.variables['lat'][21600:36015:15]
elev = topo.variables['elevation'][21600:36015:15, 24000:43215:15]
elev = np.ma.masked_where(elev > 0, elev)

ax1.set_extent([-80, 0, 0, 60], crs=ccrs.PlateCarree())
ax1.coastlines(resolution='10m', linewidth=0.8)
ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
ax1.add_feature(cfeature.LAND, facecolor='gray')
gl = ax1.gridlines(draw_labels=True, linewidth=0.5, alpha=0.6, linestyle='--')
gl.top_labels = False
gl.right_labels = False
pcm = ax1.pcolormesh(lon_topo, lat_topo, elev/1000, transform=ccrs.PlateCarree(), cmap=cmocean.cm.deep_r, vmin=-6, vmax=0, rasterized=True)
cbar = plt.colorbar(pcm, ax=ax1, orientation='vertical', pad=0.02, label='Depth (km)')
ax1.scatter(-32.75, 36.23, 20, 'r', 'x')
ax1.set_ylabel('Latitude')
ax1.text(-80, 0, 'a', ha='left', va='bottom', fontsize=14, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

ax2 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
aviso = nc.Dataset(r'L:/graduation proj/data/AVISO/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1M-m_20150901-20160901_0--80_0-60_monthly.nc')
sla = aviso.variables['sla'][:]
sla_mean = sla[0, :, :]
lon = aviso.variables['longitude'][:]
lat = aviso.variables['latitude'][:]
ax2.set_extent([-80, 0, 0, 60], crs=ccrs.PlateCarree())
ax2.coastlines(resolution='10m', linewidth=0.8)
ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
ax2.add_feature(cfeature.LAND, facecolor='gray')
gl = ax2.gridlines(draw_labels=True, linewidth=0.5, alpha=0.6, linestyle='--')
gl.top_labels = False
gl.right_labels = False
pcm = ax2.pcolormesh(lon, lat, sla_mean, transform=ccrs.PlateCarree(), cmap=cmocean.cm.balance, vmin=-0.6, vmax=0.6, rasterized=True)
cbar = plt.colorbar(pcm, ax=ax2, orientation='vertical', pad=0.02, label='SLA (m)')
ax2.scatter(-32.75, 36.23, 20, 'r', 'x')
ax2.text(-80, 0, 'b', ha='left', va='bottom', fontsize=14, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

ax3 = fig.add_subplot(gs[2], projection=ccrs.PlateCarree())
woa23 = nc.Dataset(r'L:/graduation proj/data/WOA23/woa23_B5C2_t00_04.nc')
temp = woa23.variables['t_an'][:]
lon_woa = woa23.variables['lon'][:]
lat_woa = woa23.variables['lat'][:]
lonIdx1 = np.argmin(np.abs(lon_woa-(-80)))
lonIdx2 = np.argmin(np.abs(lon_woa-0))
latIdx1 = np.argmin(np.abs(lat_woa-0))
latIdx2 = np.argmin(np.abs(lat_woa-60))
lon = lon_woa[lonIdx1:lonIdx2]
lat = lat_woa[latIdx1:latIdx2]
temp = temp[0, 0, latIdx1:latIdx2, lonIdx1:lonIdx2]
ax3.set_extent([-80, 0, 0, 60], crs=ccrs.PlateCarree())
ax3.coastlines(resolution='10m', linewidth=0.8)
ax3.add_feature(cfeature.BORDERS, linewidth=0.5)
ax3.add_feature(cfeature.LAND, facecolor='gray')
gl = ax3.gridlines(draw_labels=True, linewidth=0.5, alpha=0.6, linestyle='--')
gl.top_labels = False
gl.right_labels = False
pcm = ax3.pcolormesh(lon, lat, temp, cmap=cmocean.cm.thermal, vmin=5, vmax=30, rasterized=True)
cbar = plt.colorbar(pcm, ax=ax3, orientation='vertical', pad=0.02, label='Temperature (℃)')
ax3.scatter(-32.75, 36.23, 20, 'white', 'x')
ax3.set_ylabel('Latitude')
ax3.set_xlabel('Longitude')
ax3.text(-80, 0, 'c', ha='left', va='bottom', fontsize=14, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

ax4 = fig.add_subplot(gs[3], projection=ccrs.PlateCarree())
era5 = nc.Dataset(r'L:/graduation proj/data/ERA5/1af5513fb616a391efc348ff1ee624bb.nc')
lon = era5.variables['longitude'][:]
lat = era5.variables['latitude'][:]
u10 = era5.variables['u10'][:]
v10 = era5.variables['v10'][:]
rhoAir = 1.3
dragCoeff = 1.5e-3
tau = rhoAir * dragCoeff * (u10 ** 2 + v10 ** 2)
tau_mean = np.nanmean(tau, 0)
ax4.set_extent([-80, 0, 0, 60], crs=ccrs.PlateCarree())
ax4.coastlines(resolution='10m', linewidth=0.8, zorder=3)
ax4.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=3)
ax4.add_feature(cfeature.LAND, facecolor='gray', zorder=2)
gl = ax4.gridlines(draw_labels=True, linewidth=0.5, alpha=0.6, linestyle='--', zorder=3)
gl.top_labels = False
gl.right_labels = False
pcm = ax4.pcolormesh(lon, lat, tau_mean, cmap=cmocean.cm.speed, zorder=1, vmin=0, vmax=0.15, rasterized=True)
cbar = plt.colorbar(pcm, ax=ax4, orientation='vertical', pad=0.02, label=r'$\tau$ $(N/m^{2})$')
ax4.scatter(-32.75, 36.23, 20, 'r', 'x')
ax4.set_xlabel('Longitude')
ax4.text(-80, 0, 'd', ha='left', va='bottom', fontsize=14, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))


plt.savefig('figures/fig_1_BasicParams.jpg', dpi=300, bbox_inches='tight')
plt.savefig('figuresFinal/fig_1_BasicParams.png', dpi=300, bbox_inches='tight')
plt.savefig('figuresFinal/fig_1_BasicParams.pdf', bbox_inches='tight')
plt.show()
print('c')
