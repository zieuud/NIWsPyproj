import numpy as np
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import cmocean


fig = plt.figure(figsize=(15, 6))
gs = fig.add_gridspec(1, 2, height_ratios=[1], width_ratios=[4, 1], hspace=2, wspace=0.05)

ax1 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
topo = nc.Dataset(r'L:/graduation proj/data/GEBCO/GEBCO_2024.nc')
lon_topo = topo.variables['lon'][33120:37455]
lat_topo = topo.variables['lat'][28800:31695]
elev = topo.variables['elevation'][28800:31695, 33120:37455]/1000
elev = np.ma.masked_where(elev > 0, elev)

ax1.set_extent([-24, -42, 30, 42], crs=ccrs.PlateCarree())
ax1.coastlines(resolution='10m', linewidth=0.8)
ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
ax1.add_feature(cfeature.LAND, facecolor='gray')
gl = ax1.gridlines(draw_labels=True, linewidth=0.5, alpha=0.6, linestyle='--')
gl.top_labels = False
gl.right_labels = False
pcm = ax1.contourf(lon_topo, lat_topo, elev, transform=ccrs.PlateCarree(), cmap=cmocean.cm.deep_r, vmin=-6, vmax=0, rasterized=True)
cbar = plt.colorbar(pcm, ax=ax1, orientation='vertical', pad=0.02, label='Depth (km)')
ax1.scatter(-32.75, 36.23, 60, 'r', 'x')
ax1.set_ylabel('Latitude')
ax1.text(-42, 30, 'a', ha='left', va='bottom', fontsize=14, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

ax2 = fig.add_subplot(gs[1])
fig.subplots_adjust(bottom=0.15)
sensor = np.load(r'MoorData/SENSOR_temp.npz')
depthSensor = sensor['depth_sensor']/1000
moor = np.load(r'MoorData/ADCP_uv.npz')
depthMoor = moor['depth_adcp']/1000
u = moor['u'][0, :]
v = moor['v'][0, :]
# nan location: 40-42    118 119    169-175
ax2.scatter(np.ones(len(depthSensor)), depthSensor, 20, '#D46A6A', 'o', edgecolors='k', label='Thermometer')
ax2.plot(np.ones(118)*2, depthMoor[:118], c='#5D84A7', linestyle='-')
ax2.plot([1.95, 2.05], [depthMoor[0], depthMoor[0]], c='#5D84A7', linestyle='-')
ax2.plot([1.95, 2.05], [depthMoor[117], depthMoor[117]], c='#5D84A7', linestyle='-')
ax2.scatter(2, depthMoor[39], 20, '#B7CDE6', 'v', edgecolors='k', zorder=2, label='Upward-looking ADCP')
ax2.scatter(2, depthMoor[43], 20, '#B7CDE6', '^', edgecolors='k', zorder=2, label='Downward-looking ADCP')
ax2.plot(np.ones(125)*2, depthMoor[120:], c='#5D84A7', linestyle='-')
ax2.plot([1.95, 2.05], [depthMoor[120], depthMoor[120]], c='#5D84A7', linestyle='-')
ax2.plot([1.95, 2.05], [depthMoor[-1], depthMoor[-1]], c='#5D84A7', linestyle='-')
ax2.scatter(2, depthMoor[170], 20, '#B7CDE6', 'v', edgecolors='k', zorder=2)
ax2.scatter(2, depthMoor[174], 20, '#B7CDE6', '^', edgecolors='k', zorder=2)
ax2.text(0, depthSensor[-1], 'sea floor ', ha='right', va='center')
ax2.text(3, depthSensor[-1], 'b', ha='right', va='bottom', fontsize=14)

ax2.set_xlim([0, 3])
ax2.set_ylim([depthSensor[-1], 0])
ax2.set_xticks([])
ax2.set_ylabel('Depth (km)')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=1)
ax2.spines['top'].set_visible(False)    # 去掉上边框
ax2.spines['right'].set_visible(False)  # 去掉右边框
ax2.tick_params(direction='in')

plt.savefig('figures/fig_2_MooringSetting.jpg', dpi=300, bbox_inches='tight')
plt.savefig('figuresFinal/fig_2_MooringSetting.png', dpi=300, bbox_inches='tight')
plt.savefig('figuresFinal/fig_2_MooringSetting.pdf', bbox_inches='tight')
plt.show()
print('c')
