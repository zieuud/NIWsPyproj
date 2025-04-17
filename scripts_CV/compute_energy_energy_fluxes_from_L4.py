'''
CV 2020/01/23: compute kinetic energy density and energy fluxes from L4 for all moorings
               no modal decomposition is done, have a look at compute_energy_energy_fluxes_from_L3_modes.py if interested
CV 2020/08/21: repeat with different filtering parameters 
''' 
from rrex_moorings import RREX_moorings
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
import matplotlib.colors as colors
import matplotlib.dates  as dates
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import scipy.signal as sig
import scipy.interpolate as itp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import gsw as gsw
from netCDF4 import Dataset

# ------------ parameters ------------ 
path_data = '/Users/cv1m15/Data/'
#ti,tf = 23920,23920+360 # first file 
ti,tf = 23920+360,24674 # second file  
#ti,tf = 23920,24674  # works for all moorings except ICW
seafloor = np.asarray([2063,1478,1467,1449,2110,2098,2302]) # geospatial_vertical_max in original files 
g        = 9.81
c        = 1.105 # filtering parameter 

#file_out = path_data+'energy_energy_fluxes_all_moorings_1.nc' 
#file_out = path_data+'energy_energy_fluxes_all_moorings_2.nc' 
#file_out = path_data+'energy_energy_fluxes_all_moorings_c1dot035_1.nc' 
#file_out = path_data+'energy_energy_fluxes_all_moorings_c1dot035_2.nc' 
#file_out = path_data+'energy_energy_fluxes_all_moorings_c1dot105_1.nc' 
file_out = path_data+'energy_energy_fluxes_all_moorings_c1dot105_2.nc' 


# ------------ read data ------------
data = RREX_moorings(time=[ti,tf]) # no list --> all moorings, level=L4
data.get_date()

# ------------ processing ------------ 
print(' ... processing ... ')
print('######################## 1. COMPUTE VELOCITY PERTURBATION ##############################')
# --- remove low-pass velocity --- 
u_lp = data.time_filtering(data.u,filter_type='lp')
v_lp = data.time_filtering(data.v,filter_type='lp')
up = data.u - u_lp # u prime 
vp = data.v - v_lp
 
# --- baroclinicity condition --- 
ze = 0.5*(data.zc[1:]+data.zc[:-1])
dz = ze[1]-ze[0]
seafloor_tile = np.tile(seafloor,(data.nt,1)).T
up = up - np.transpose(np.tile(np.nanmean(up*dz,axis=-1)/seafloor_tile,(data.nz,1,1)),(1,2,0))
vp = vp - np.transpose(np.tile(np.nanmean(vp*dz,axis=-1)/seafloor_tile,(data.nz,1,1)),(1,2,0))

# --- filtering --- 
u_sd = data.time_filtering(up,filter_type='sd',c=c)
v_sd = data.time_filtering(vp,filter_type='sd',c=c)
u_ni = data.time_filtering(up,filter_type='ni',c=c)
v_ni = data.time_filtering(vp,filter_type='ni',c=c)

print('######################## 2. COMPUTE PRESSURE PERTURBATION ##############################')
# --- remove low-pass temperature --- 
t_lp  = data.time_filtering(data.t,filter_type='lp') # low-pass 
tp = data.t - t_lp

####################### METHOD 1: TEMPERATURE GRADIENT FROM IN SITU DATA ############################
# --- compute temperature gradient from very low-pass data --- 
#t_vlp  = data.time_filtering(data.t,filter_type='vlp') # very low-pass
#dtdz_e = np.diff(t_vlp,axis=2)/dz 
#dtdz_c = np.zeros((data.nm,data.nt,data.nz)) 
#dtdz_c[:,:,1:-1] = 0.5*(dtdz_e[:,:,1:]+dtdz_e[:,:,:-1])  
#dtdz_c[dtdz_c<1e-4] = 1e-4 # lower bound in very weakly stratified waters where dT/dz goes to zero

####################### METHOD 2: TEMPERATURE GRADIENT FROM CLIMATOLOGICAL DATA #####################
data.get_N2_from_WOA()
# --- compute displacement ---
#x = -tp/dtdz_c             # --> from gradient computed using thermistors' data  
x = -tp/data.dtdz_woa_grid  # --> from gradient computed with climatological data 

#plt.figure()
#plt.contourf(data.time,data.zc,x[0,:,:].T,20);plt.colorbar() 
#plt.savefig('tmp.png')

# --- filtering --- 
x_sd = data.time_filtering(x,filter_type='sd',c=c)
x_ni = data.time_filtering(x,filter_type='ni',c=c)

# --- compute pressure anomaly --- 
rhoa    = ((data.sig0_woa_grid+1000)/g)*data.N2_woa_grid*x
rhoa_sd = ((data.sig0_woa_grid+1000)/g)*data.N2_woa_grid*x_sd
rhoa_ni = ((data.sig0_woa_grid+1000)/g)*data.N2_woa_grid*x_ni

p     = np.zeros((data.nm,data.nt,data.nz))
p_sd  = np.zeros((data.nm,data.nt,data.nz))
p_ni  = np.zeros((data.nm,data.nt,data.nz))

for k in range(data.nz):
    p[:,:,k]    = np.nansum(rhoa[:,:,k:]*dz*g,axis=2) 
    p_sd[:,:,k] = np.nansum(rhoa_sd[:,:,k:]*dz*g,axis=2) 
    p_ni[:,:,k] = np.nansum(rhoa_ni[:,:,k:]*dz*g,axis=2) 

# --- baroclinicity condition --- 
pbar    = np.transpose(np.tile(np.nansum(p,axis=-1)/seafloor_tile,(data.nz,1,1)),(1,2,0))
pbar_sd = np.transpose(np.tile(np.nansum(p_sd,axis=-1)/seafloor_tile,(data.nz,1,1)),(1,2,0))
pbar_ni = np.transpose(np.tile(np.nansum(p_ni,axis=-1)/seafloor_tile,(data.nz,1,1)),(1,2,0))
p    = p - pbar 
p_sd = p_sd - pbar_sd
p_ni = p_ni - pbar_ni

#plt.figure()
#plt.subplot(211);plt.contourf(data.time,data.zc,p_sd[0,:,:].T,20);plt.colorbar() 
#plt.subplot(212);plt.contourf(data.time,data.zc,p_ni[0,:,:].T,20);plt.colorbar() 
#plt.savefig('tmp.png')

print('######################## 3. SAVE IN NETCDF FILE ########################################')
nc = Dataset(file_out,'w')
nc.createDimension('m',data.nm)
nc.createDimension('t',data.nt)
nc.createDimension('z',data.nz) 
# - grid - 
nc.createVariable('time','f',('t')) 
nc.createVariable('z','f',('z')) 
nc.variables['time'][:] = data.time
nc.variables['z'][:]    = data.zc 
# - u - 
nc.createVariable('up','f',('m','t','z')) 
nc.createVariable('u_sd','f',('m','t','z')) 
nc.createVariable('u_ni','f',('m','t','z')) 
nc.variables['up'][:]   = up 
nc.variables['u_sd'][:] = u_sd
nc.variables['u_ni'][:] = u_ni
# - v - 
nc.createVariable('vp','f',('m','t','z')) 
nc.createVariable('v_sd','f',('m','t','z')) 
nc.createVariable('v_ni','f',('m','t','z')) 
nc.variables['vp'][:]   = vp 
nc.variables['v_sd'][:] = v_sd
nc.variables['v_ni'][:] = v_ni
# - p - 
nc.createVariable('p','f',('m','t','z')) 
nc.createVariable('p_sd','f',('m','t','z')) 
nc.createVariable('p_ni','f',('m','t','z')) 
nc.variables['p'][:]    = p 
nc.variables['p_sd'][:] = p_sd
nc.variables['p_ni'][:] = p_ni
nc.close()  

