'''
CV 2019/09/11: create a class with functions associated to mooring processing 
''' 
import numpy as np
from netCDF4 import Dataset
import scipy.io as sio
import scipy.stats as stats
import scipy.interpolate as itp
import scipy.signal as sig
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
import time as tm
import datetime as datetime 
import gsw as gsw
import os
import sys
sys.path.append('../../../Python_miscellaneous/')
from distance_sphere_matproof import dist_sphere
import warnings
warnings.filterwarnings('ignore')

def round_time(dt=None, round_to=60):
    # https://kite.com/python/examples/4653/datetime-round-datetime-to-any-time-interval-in-seconds
    if dt == None: 
       dt = datetime.datetime.now()
    seconds = (dt - dt.min).seconds
    rounding = (seconds+round_to/2) // round_to * round_to
    return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)

M2    = 1./44700. # [sec-1] 
S2    = 1./43200.
seafloor = np.asarray([2063,1478,1467,1449,2110,2098,2302]) # geospatial_vertical_max in original files

class RREX_moorings(object): 
    def __init__(self,**kwargs): 
        print(' ------------ initiate a class with RREX moorings data and methods ------------ ') 
        self.path_data     = '/Users/cv1m15/Data/'
        self.file_L4       = '_gridded.nc' 
        self.file_all      = 'all_moorings_gridded.nc' 
        self.file_topo_lr  = self.path_data+'ETOPO2v2c_f4.nc' 
        self.file_topo_hr  = self.path_data+'topo15_NorthAtl.nc' # chunk of SRTM15-PLUS
        self.file_wind     = self.path_data+'era5_rrex_moorings.nc'
        self.file_energy   = self.path_data+'energy_energy_fluxes_all_moorings.nc' 
        #self.file_aviso    = self.path_data+'Aviso/dt_global_allsat_phy_l4_'
        self.file_aviso    = '/Volumes/cvic_usb/Aviso/dt_global_allsat_phy_l4_'
        #self.file_otps     = '/Users/cv1m15/Documents/OSU_Tidal_Software/OTPS/tides_otps_moorings.nc' 
        self.file_otps     = self.path_data+'tides_otps_moorings.nc' 
        self.file_tpxo     = self.path_data+'tpxo9_spring-neap_northatl.nc'  
        self.list_moorings = kwargs.get('list_moorings',[])
        self.nm            = len(self.list_moorings)
        [ti,tf]            = kwargs.get('time',[23920,24674])
        self.level         = kwargs.get('level','L4') 
        if (ti<23920) or (tf>24674): 
            exit('ERROR: time boundaries are off the acceptable limits') 
        if self.nm==0: 
            print(' --> no list of moorings provided, guess you want to load them all ')
            self.list_moorings = ['IRW','IRM','IRE','RRT','ICW','ICM','ICE'] 
            self.nm            = len(self.list_moorings) # update number of moorings 
        else: 
            print(' --> load data from moorings: ')
            print(*self.list_moorings)
        ################### CASE: SINGLE MOORING NOT INTERPOLATED IN Z ###########
        if self.nm==1 and self.level=='L3': 
            nc = Dataset(self.path_data+'RREX_'+self.list_moorings[0]+'_CURR_hourly.nc','r') 
            time_tmp  = nc.variables['TIME'][:]  
            tmin      = np.argwhere(time_tmp==ti)[0][0]
            tmax      = np.argwhere(time_tmp==tf)[0][0]
            self.time = nc.variables['TIME'][tmin:tmax+1]
            self.u    = nc.variables['UCUR'][tmin:tmax+1,:] 
            self.v    = nc.variables['VCUR'][tmin:tmax+1,:] 
            self.z_uv = -nc.variables['DEPTH'][tmin:tmax+1,:]
            self.lat  = nc.variables['LATITUDE'][0]
            self.lon  = nc.variables['LONGITUDE'][0]
            u_qc      = nc.variables['UCUR_QC'][tmin:tmax+1,:]
            v_qc      = nc.variables['VCUR_QC'][tmin:tmax+1,:]
            nc.close() 
            nc = Dataset(self.path_data+'RREX_'+self.list_moorings[0]+'_TS_hourly.nc','r')
            self.t    = nc.variables['TEMP'][tmin:tmax+1,:] # assume same time line as in the CURR file 
            self.s    = nc.variables['PSAL'][tmin:tmax+1,:]
            self.z_ts = -nc.variables['DEPTH'][tmin:tmax+1,:]
            t_qc      = nc.variables['TEMP_QC'][tmin:tmax+1,:] 
            s_qc      = nc.variables['PSAL_QC'][tmin:tmax+1,:] 
            nc.close() 
            self.nt,self.nz_uv = self.u.shape 
            self.nt,self.nz_ts = self.t.shape
            # - nan bad data -  
            self.t[t_qc==3]=np.nan; self.t[t_qc==4]=np.nan; self.t[t_qc==9]=np.nan
            self.s[s_qc==3]=np.nan; self.s[s_qc==4]=np.nan; self.s[s_qc==9]=np.nan
            self.u[u_qc==3]=np.nan; self.u[u_qc==4]=np.nan; self.u[u_qc==9]=np.nan
            self.v[v_qc==3]=np.nan; self.v[v_qc==4]=np.nan; self.v[v_qc==9]=np.nan
        
        ################### CASE: SINGLE MOORING INTERPOLATED IN Z ###############
        #elif self.nm==1 and self.level=='L4': 
        #    nc = Dataset(self.path_data+self.list_moorings[0]+self.file_L4,'r')
        #    time_tmp  = nc.variables['time'][:]  
        #    tmin      = np.argwhere(time_tmp==ti)[0][0]
        #    tmax      = np.argwhere(time_tmp==tf)[0][0]
        #    print(tmin,tmax) 
        #    self.time = nc.variables['time'][tmin:tmax+1]
        #    self.zc   = nc.variables['z'][:]
        #    self.u    = nc.variables['u'][tmin:tmax+1,:]  
        #    self.v    = nc.variables['v'][tmin:tmax+1,:]  
        #    self.t    = nc.variables['T'][tmin:tmax+1,:]  
        #    self.s    = nc.variables['S'][tmin:tmax+1,:]
        #    self.z_uv = nc.variables['z_uv'][tmin:tmax+1,:]
        #    self.lat  = nc.lat 
        #    self.lon  = nc.lon 
        #    nc.close()
        #    self.nt,self.nz = self.u.shape
        #    self.nz_uv = self.nz # for script compatibility  
        ################### CASE: MULTIPLE MOORINGS ##############################
        else:
            if not os.path.isfile(self.path_data+self.file_all): 
                # ------ interpolate to common grid and save to netcdf file ------ 
                ############################################################################# 
                # NB: THIS HAS TO BE DONE ONLY ONCE WITH ti=23920, tf=24674 AND ALL MOORINGS
                ############################################################################# 
                print(' ... get the common grid dimension ... ')  
                zmin = 0 # max depth
                n_uv = 0 # max independent measurements of u,v  
                n_ts = 0 # max independent measurements of T,S  
                for m in range(self.nm):
                    nc = Dataset(self.path_data+self.list_moorings[m]+self.file_L4,'r')
                    zmin = np.nanmin((zmin,np.nanmin(nc.variables['z'][:])))
                    n_uv = np.nanmax((n_uv,nc.variables['z_uv'][:].shape[1]))  
                    n_ts = np.nanmax((n_ts,nc.variables['z_ts'][:].shape[1]))  
                    if m==0:
                        dz = nc.variables['z'][1]-nc.variables['z'][0] # assumed to be the same in all files
                        time = nc.variables['time'][:]  
                        tmin = np.argwhere(time==ti)[0][0]
                        tmax = np.argwhere(time==tf)[0][0]
                        self.nt = tmax-tmin+1 
                    nc.close()
                self.zc = np.arange(zmin,0,dz)
                self.nz = self.z.shape[0] 
                self.u = np.zeros((self.nm,self.nt,self.nz)) 
                self.v = np.zeros((self.nm,self.nt,self.nz)) 
                self.t = np.zeros((self.nm,self.nt,self.nz)) 
                self.s = np.zeros((self.nm,self.nt,self.nz)) 
                self.lat = np.zeros(self.nm) 
                self.lon = np.zeros(self.nm)
                self.z_uv = np.zeros((self.nm,self.nt,n_uv)) 
                self.z_ts = np.zeros((self.nm,self.nt,n_ts)) 
                for m in range(self.nm):
                    print(' ... interpolate data for mooring '+self.list_moorings[m]+' ... ') 
                    nc = Dataset(self.path_data+self.list_moorings[m]+self.file_L4,'r')
                    self.lon[m] = nc.lon 
                    self.lat[m] = nc.lat 
                    time = nc.variables['time'][:] 
                    tmin = np.argwhere(time==ti)[0][0]
                    tmax = np.argwhere(time==tf)[0][0]
                    um   = nc.variables['u'][tmin:tmax+1,:] 
                    vm   = nc.variables['v'][tmin:tmax+1,:] 
                    Tm   = nc.variables['T'][tmin:tmax+1,:] 
                    Sm   = nc.variables['S'][tmin:tmax+1,:] 
                    zm   = nc.variables['z'][:] 
                    nloc_uv                 = nc.variables['z_uv'][:].shape[1]  # local dimension 
                    self.z_uv[m,:,:nloc_uv] = nc.variables['z_uv'][tmin:tmax+1,:]
                    nloc_ts                 = nc.variables['z_ts'][:].shape[1] 
                    self.z_ts[m,:,:nloc_ts] = nc.variables['z_ts'][tmin:tmax+1,:]
                    nc.close() 
                    self.time = time[tmin:tmax+1] 
                    for t in range(self.nt):
                        tmp    = um[t,:] 
                        interp = itp.interp1d(zm[~np.isnan(tmp)],tmp[~np.isnan(tmp)],bounds_error=False)
                        self.u[m,t,:] = interp(self.z) 
                        tmp    = vm[t,:] 
                        interp = itp.interp1d(zm[~np.isnan(tmp)],tmp[~np.isnan(tmp)],bounds_error=False)
                        self.v[m,t,:] = interp(self.z) 
                        tmp    = Tm[t,:] 
                        interp = itp.interp1d(zm[~np.isnan(tmp)],tmp[~np.isnan(tmp)],bounds_error=False)
                        self.t[m,t,:] = interp(self.z) 
                        tmp    = Sm[t,:] 
                        interp = itp.interp1d(zm[~np.isnan(tmp)],tmp[~np.isnan(tmp)],bounds_error=False)
                        self.s[m,t,:] = interp(self.z) 
                        # - quick check -  
                        #plt.figure()
                        #plt.plot(tmp,zm,'k',lw=0.6) 
                        #plt.plot(u[m,t,:],z,'r',lw=0.3) 
                        #plt.savefig('tmp.pdf'); exit()
                nc = Dataset(self.path_data+self.file_all,'w')
                nc.list_moorings = self.list_moorings 
                nc.createDimension('mooring',self.nm)
                nc.createDimension('time',self.nt)
                nc.createDimension('z',self.nz)
                nc.createDimension('n_uv',n_uv)
                nc.createDimension('n_ts',n_ts)
                nc.createVariable('lat','f',('mooring',))
                nc.createVariable('lon','f',('mooring',))
                nc.createVariable('time','d',('time',))
                nc.createVariable('z','f',('z',))
                nc.createVariable('u','f',('mooring','time','z'))
                nc.createVariable('v','f',('mooring','time','z'))
                nc.createVariable('T','f',('mooring','time','z'))
                nc.createVariable('S','f',('mooring','time','z'))
                nc.createVariable('z_uv','f',('mooring','time','n_uv'))
                nc.createVariable('z_ts','f',('mooring','time','n_ts'))
                nc.variables['lon'][:] = self.lon 
                nc.variables['lat'][:] = self.lat 
                nc.variables['time'][:] = self.time
                nc.variables['z'][:] = self.z
                nc.variables['u'][:] = self.u
                nc.variables['v'][:] = self.v
                nc.variables['T'][:] = self.t
                nc.variables['S'][:] = self.s
                nc.variables['z_uv'][:] = self.z_uv
                nc.variables['z_ts'][:] = self.z_ts
                nc.close()
  
            else: # file with all mooring data exists 
                nc     = Dataset(self.path_data+self.file_all,'r') 
                ind_moor = [] 
                for m in range(len(self.list_moorings)): 
                    ind_moor.append(nc.list_moorings.index(self.list_moorings[m])) 
                self.lat  = nc.variables['lat'][ind_moor] 
                self.lon  = nc.variables['lon'][ind_moor] 
                time      = nc.variables['time'][:] 
                tmin      = np.argwhere(time==ti)[0][0]
                tmax      = np.argwhere(time==tf)[0][0]
                self.u    = np.squeeze(nc.variables['u'][ind_moor,tmin:tmax+1,:]) 
                self.v    = np.squeeze(nc.variables['v'][ind_moor,tmin:tmax+1,:]) 
                self.t    = np.squeeze(nc.variables['T'][ind_moor,tmin:tmax+1,:]) 
                self.s    = np.squeeze(nc.variables['S'][ind_moor,tmin:tmax+1,:]) 
                self.zc   = nc.variables['z'][:]
                self.z_uv = np.squeeze(nc.variables['z_uv'][ind_moor,tmin:tmax+1,:])  
                self.z_ts = np.squeeze(nc.variables['z_ts'][ind_moor,tmin:tmax+1,:])  
                nc.close() 
                self.time = time[tmin:tmax+1] 
                self.nz   = self.zc.shape[0]  
                self.nt   = self.time.shape[0]  
                self.seafloor = seafloor[ind_moor] 
        return

    def get_tpxo(self): 
        print(' ... get TPXO tidal current parameters ... ')
        nc    = Dataset(self.file_tpxo,'r') 
        lon   = nc.variables['lon'][:] 
        lat   = nc.variables['lat'][:] 
        ua_s  = nc.variables['ua_spring'][:]  
        up_s  = nc.variables['up_spring'][:]  
        va_s  = nc.variables['va_spring'][:]  
        vp_s  = nc.variables['vp_spring'][:]  
        ua_m2 = nc.variables['ua_m2'][:]  
        up_m2 = nc.variables['up_m2'][:]  
        va_m2 = nc.variables['va_m2'][:]  
        vp_m2 = nc.variables['vp_m2'][:]  
        # - interpolate to mooring location - 
        spline     = itp.RectBivariateSpline(lat,lon,ua_s,kx=1,ky=1)
        self.ua_s  = spline.ev(self.lat,self.lon)
        spline     = itp.RectBivariateSpline(lat,lon,up_s,kx=1,ky=1)
        self.up_s  = spline.ev(self.lat,self.lon)
        spline     = itp.RectBivariateSpline(lat,lon,va_s,kx=1,ky=1)
        self.va_s  = spline.ev(self.lat,self.lon)
        spline     = itp.RectBivariateSpline(lat,lon,vp_s,kx=1,ky=1)
        self.vp_s  = spline.ev(self.lat,self.lon)
        spline     = itp.RectBivariateSpline(lat,lon,ua_m2,kx=1,ky=1)
        self.ua_m2 = spline.ev(self.lat,self.lon)
        spline     = itp.RectBivariateSpline(lat,lon,up_m2,kx=1,ky=1)
        self.up_m2 = spline.ev(self.lat,self.lon)
        spline     = itp.RectBivariateSpline(lat,lon,va_m2,kx=1,ky=1)
        self.va_m2 = spline.ev(self.lat,self.lon)
        spline     = itp.RectBivariateSpline(lat,lon,vp_m2,kx=1,ky=1)
        self.vp_m2 = spline.ev(self.lat,self.lon)
        return  

    def get_otps_tides(self):
        print(' ... get OTPS tidal currents ... ')
        nc = Dataset(self.file_otps,'r') 
        time = nc.variables['time'][:] 
        tmin = np.argwhere(time==self.time[0])[0][0]
        tmax = np.argwhere(time==self.time[-1])[0][0]
        ind_moor = []
        list_moorings = ['IRW','IRM','IRE','RRT','ICW','ICM','ICE']
        for m in range(len(self.list_moorings)):
            ind_moor.append(list_moorings.index(self.list_moorings[m])) 
        self.u_otps = np.squeeze(nc.variables['u'][ind_moor,:,tmin:tmax+1]) 
        self.v_otps = np.squeeze(nc.variables['v'][ind_moor,:,tmin:tmax+1]) 
        return 

    def get_aviso(self,lonmin=-60,lonmax=-10,latmin=45,latmax=65,itp_hourly=False,itp_moor=False):
        print(' ... get Aviso fields ... ')
        jul_beg = int(np.nanmin(self.time))   # first index of file to load 
        jul_end = int(np.nanmax(self.time))+1 # last  index of file to load
        imin, imax, jmin, jmax = 0,0,0,0
        # - get boundaries of domain to load - 
        nc  = Dataset(self.file_aviso+'20150601_20170110.nc','r') 
        lon = nc.variables['longitude'][:]; lon[lon>180.]-=360.
        lat = nc.variables['latitude'][:]
        fillval = nc.variables['adt']._FillValue
        nc.close()
        imin = np.argmin(abs(lon-lonmin))-1
        imax = np.argmin(abs(lon-lonmax))+1
        jmin = np.argmin(abs(lat-latmin))-1
        jmax = np.argmin(abs(lat-latmax))+1
        self.lonavi = lon[imin:imax]
        self.latavi = lat[jmin:jmax]
        # - read fields - 
        print('      --> read daily fields ') 
        self.adt        = np.zeros((jul_end-jul_beg+1,jmax-jmin,imax-imin))
        self.ugos       = np.zeros((jul_end-jul_beg+1,jmax-jmin,imax-imin))
        self.vgos       = np.zeros((jul_end-jul_beg+1,jmax-jmin,imax-imin))
        self.time_aviso = np.zeros((jul_end-jul_beg+1,))
        diff_days = (datetime.datetime(1950,1,1) - datetime.datetime(1970,1,1)).days # offset in CTD/AVISO compared to Python 
        timeline = [tm.gmtime((i+diff_days)*86400) for i in np.arange(jul_beg,jul_end+1,1)]
        process_time_list = ['20170110','20180516','20170209','20170530','20170914']
        for i in range(jul_end-jul_beg+1):
            datestr = '%.4i%.2i%.2i'%(timeline[i][0],timeline[i][1],timeline[i][2])
            file_list = [self.file_aviso+datestr+'_'+process_time+'.nc' for process_time in process_time_list]
            j = 0
            while not os.path.isfile(file_list[j]):j+=1 
            nc = Dataset(file_list[j],'r')
            self.adt[i,:,:]    = nc.variables['adt'][:,jmin:jmax,imin:imax].data
            self.ugos[i,:,:]   = nc.variables['ugos'][:,jmin:jmax,imin:imax].data
            self.vgos[i,:,:]   = nc.variables['vgos'][:,jmin:jmax,imin:imax].data
            self.time_aviso[i] = nc.variables['time'][:].data
            nc.close()
        self.nt_aviso = self.adt.shape[0]
        self.ymdhms_aviso = [datetime.datetime(timeline[i][0],timeline[i][1],timeline[i][2],
                                               timeline[i][3],timeline[i][4],timeline[i][5]) 
                             for i in np.arange(self.nt_aviso)]
        # - now interpolate to mooring timeline -
        if itp_hourly: 
            print('      --> interpolate hourly ') 
            interp_adt       = itp.interp1d(self.time_aviso,adt,axis=0)
            self.adt_hourly  = interp_adt(self.time.data)
            interp_ugos      = itp.interp1d(self.time_aviso,ugos,axis=0)
            self.ugos_hourly = interp_adt(self.time.data)
            interp_vgos      = itp.interp1d(self.time_aviso,vgos,axis=0)
            self.vgos_hourly = interp_adt(self.time.data)
            # - mask data - 
            self.adt[self.adt   == fillval] = 0 # not nan to enable interpolations      
            self.ugos[self.ugos == fillval] = 0
            self.vgos[self.vgos == fillval] = 0 
        # - interpolate at mooring locations -
        if itp_moor:
            print('      --> interpolate at mooring locations ') 
            self.adt_moor  = np.zeros((self.nm,self.nt))
            self.ugos_moor = np.zeros((self.nm,self.nt))
            self.vgos_moor = np.zeros((self.nm,self.nt))
            for t in range(self.nt):
                spline = itp.RectBivariateSpline(self.lonavi,self.latavi,self.adt_hourly[t,:,:].T,kx=1,ky=1)
                self.adt_moor[:,t] = spline.ev(self.lon,self.lat)
                spline = itp.RectBivariateSpline(self.lonavi,self.latavi,self.ugos_hourly[t,:,:].T,kx=1,ky=1)
                self.ugos_moor[:,t] = spline.ev(self.lon,self.lat)
                spline = itp.RectBivariateSpline(self.lonavi,self.latavi,self.vgos_hourly[t,:,:].T,kx=1,ky=1)
                self.vgos_moor[:,t] = spline.ev(self.lon,self.lat)
        return 

    def get_energy_density_fluxes(self,ke=True,flux=True,c=1.07): 
        print(' ... get energy and/or energy fluxes ... ') 
        # -------------------------------------- 
        # argument c is the filtering parameter 
        # files are available for c in [1.07,1.035,1.105]
        # default file is for c=1.07  
        # -------------------------------------- 
        if   c == 1.035:
            file_energy = self.file_energy[:-3]+'_c1dot035.nc' 
        elif c == 1.105:
            file_energy = self.file_energy[:-3]+'_c1dot105.nc' 
        else:
            file_energy = self.file_energy
        print(file_energy) 
        rho0       = 1025. 
        nc         = Dataset(file_energy,'r')
        time_tmp   = nc.variables['time'][:] 
        tmin       = np.argwhere(time_tmp==self.time[0])[0][0]
        tmax       = np.argwhere(time_tmp==self.time[-1])[0][0] 
        ind_moor = []
        list_moorings = ['IRW','IRM','IRE','RRT','ICW','ICM','ICE']
        for m in range(len(self.list_moorings)):
            ind_moor.append(list_moorings.index(self.list_moorings[m]))
        if ke:
            self.ke_sd = np.squeeze(0.5*rho0*( nc.variables['u_sd'][ind_moor,tmin:tmax+1,:]**2 
                                             + nc.variables['v_sd'][ind_moor,tmin:tmax+1,:]**2)) 
            self.ke_ni = np.squeeze(0.5*rho0*( nc.variables['u_ni'][ind_moor,tmin:tmax+1,:]**2 
                                             + nc.variables['v_ni'][ind_moor,tmin:tmax+1,:]**2)) 
        if flux: 
            self.fx_sd = np.squeeze(nc.variables['u_sd'][ind_moor,tmin:tmax+1,:]
                                   *nc.variables['p_sd'][ind_moor,tmin:tmax+1,:])
            self.fy_sd = np.squeeze(nc.variables['v_sd'][ind_moor,tmin:tmax+1,:]
                                   *nc.variables['p_sd'][ind_moor,tmin:tmax+1,:])
            self.fx_ni = np.squeeze(nc.variables['u_ni'][ind_moor,tmin:tmax+1,:]
                                   *nc.variables['p_ni'][ind_moor,tmin:tmax+1,:])
            self.fy_ni = np.squeeze(nc.variables['v_ni'][ind_moor,tmin:tmax+1,:]
                                   *nc.variables['p_ni'][ind_moor,tmin:tmax+1,:]) 
        nc.close() 
        return 

    def time_filtering(self,var,filter_type='sd',c=1.07): 
        print(' ... time filtering of the data ... ') 
        if filter_type not in ['sd','ni','iw','lp','vlp','vvlp','ms']: 
            exit('ERROR: filter_type must be in [sd,ni,si,lp,vlp], i.e., \n'\
                 '       semi-diurnal,near-inertial,internal-wave,low-pass,very low-pass')
        freq_longname={'sd':'semi-diurnal','ni':'near-inertial','iw':'internal-waves',
                       'lp':'low-pass','vlp':'very low-pass','vvlp':' very very low-pass',
                       'ms':'mesoscale'}
        print('     --> filtering variable at ',freq_longname[filter_type],' frequency')
        # --- set up filtering parameters --- 
        #c     = 1.07 # Alford's parameter is 1.25 
        Nbut  = 4 # order of Butterworth filter
        dt    = round((self.time[1]-self.time[0])*24*3600) # [sec] 
        sd    = 0.5*(M2+S2) 
        fmeso = 1./(120*86400.) # [sec-1] 
        # --- loop on moorings and depth to better handle nans --- 
        # - /!\ ACHTUNG: assumes var(mooring,time,depth) or var(time,depth) - 
        if self.nm==1: var=np.tile(var,(1,1,1)) # duplicate to avoid if statements in the loop  
        var_filt = np.copy(var)*np.nan  
        nt = var.shape[1] # could be different than self.nt in case of a derivative
        for m in range(self.nm):
            print('         * mooring %i'%m) 
            # --- design filter, depending on mooring latitude (f-related filters) ---  
            if self.nm>1: fi = gsw.f(self.lat[m])/(2*np.pi)
            else: fi = gsw.f(self.lat)/(2*np.pi)
            if filter_type=='sd': # semi-diurnal frequencies
                Wn  = np.array([(1./c)*sd,c*sd])*(2*dt)
                b,a = sig.butter(Nbut,Wn,btype='bandpass',output='ba')  
            elif filter_type=='ni': # inertial frequency  
                Wn  = np.array([(1./c)*fi,c*fi])*(2*dt)
                b,a = sig.butter(Nbut,Wn,btype='bandpass',output='ba')  
            elif filter_type=='ms': # mesoscales          
                Wn  = np.array([fmeso,(1./c)*fi])*(2*dt)
                b,a = sig.butter(Nbut,Wn,btype='bandpass',output='ba')  
            elif filter_type=='iw': # internal waves, in the super-inertial band 
                #Wn = 1*fi*(2*dt) 
                Wn = (1./c)*fi*(2*dt) 
                b,a = sig.butter(Nbut,Wn,btype='highpass',output='ba')  
            elif filter_type=='lp': # low-pass at a few f = mesoscale 
                Wn = 0.5*fi*(2*dt) 
                b,a = sig.butter(Nbut,Wn,btype='lowpass',output='ba')  
            elif filter_type=='vlp': # very low-pass (~weekly) 
                Wn = 0.1*fi*(2*dt)
                b,a = sig.butter(Nbut,Wn,btype='lowpass',output='ba')  
            elif filter_type=='vvlp': # very very low-pass (~monthly) 
                Wn = (1./(30*86400))*(2*dt)
                b,a = sig.butter(Nbut,Wn,btype='lowpass',output='ba')  
            for k in range(var.shape[-1]):  
                tmp = np.copy(var[m,:,k])  
                tmin = 0 
                if tmp[~np.isnan(tmp)].shape[0]>=10: # filter only if there is at least 10 non-nan values 
                    while tmin<nt-2:  
                        # - get intervals of non-nan data - 
                        while np.isnan(tmp[tmin]) and tmin<nt-2:tmin+=1
                        tmax = tmin+1  
                        while ~np.isnan(tmp[tmax]) and tmax<nt-1:tmax+=1 
                        if tmax-tmin>27: # ValueError: The length of the input vector x must be at least padlen, which is 27. 
                            if filter_type in ['sd','ni','ms']:
                                var_filt[m,tmin:tmax,k] = sig.filtfilt(b,a,tmp[tmin:tmax]) 
                            elif filter_type in ['iw','lp','vlp','vvlp']:
                                var_filt[m,tmin:tmax,k] = sig.filtfilt(b,a,tmp[tmin:tmax],method='gust') 
                        tmin = tmax+1
        var_filt = np.squeeze(var_filt) 
        return var_filt    
    
    def get_wind(self): 
        print(' ... get wind in the mooring area ... ')
        nc = Dataset(self.file_wind,'r') 
        self.lonw = nc.variables['lon'][:] 
        self.latw = nc.variables['lat'][:] 
        time      = nc.variables['time'][:]
        tmin      = np.argwhere(time==self.time[0])[0][0]
        tmax      = np.argwhere(time==self.time[-1])[0][0]
        self.u10  = nc.variables['u10'][tmin:tmax+1,:,:] 
        self.v10  = nc.variables['v10'][tmin:tmax+1,:,:] 

    def get_wind_moorings(self): 
        print(' ... get wind at moorings location ... ')
        nc = Dataset(self.file_wind,'r') 
        lonw  = nc.variables['lon'][:] 
        latw  = nc.variables['lat'][:] 
        time  = nc.variables['time'][:]
        tmin  = np.argwhere(time==self.time[0])[0][0]
        tmax  = np.argwhere(time==self.time[-1])[0][0]
        u10   = nc.variables['u10'][tmin:tmax+1,:,:] 
        v10   = nc.variables['v10'][tmin:tmax+1,:,:] 
        #self.time_wind = time[tmin:tmax+1]  # useless as same as self.time :-) 
        # - interpolate at moorings location -
        self.u10m = np.zeros((self.nm,self.nt)) 
        self.v10m = np.zeros((self.nm,self.nt)) 
        for t in range(tmax-tmin+1): 
            spline        = itp.RectBivariateSpline(latw,lonw,u10[t,:,:],kx=1,ky=1)
            self.u10m[:,t] = spline.ev(self.lat,self.lon) 
            spline        = itp.RectBivariateSpline(latw,lonw,v10[t,:,:],kx=1,ky=1)
            self.v10m[:,t] = spline.ev(self.lat,self.lon) 
        nc.close() 
        self.u10m = np.squeeze(self.u10m) #  in case nm=1
        self.v10m = np.squeeze(self.v10m)
        return 

    def get_Cd(self):
        print(' ... get air-sea drag coefficient ... ')
        # follows Large and Yeager - Climate Dynamics 2009 
        w10 = (self.u10m**2+self.v10m**2)**0.5
        w10[w10<0.5] = 0.5 # [m/s] to avoid infinite values of Cd, that's done in CROCO (Lionel Renault)
        a1 = 0.0027
        a2 = 0.000142
        a3 = 0.0000764
        a8 = -3.14807e-13
        Cd = a1/w10 + a2 + a3*w10 + a8*w10**6
        Cd[w10>33] = 0.00234 
        self.Cd = Cd 
        return 

    def get_dist(self):
        print(' ... get distance between moorings ... ')
        dist = dist_sphere(self.lat[:-1],self.lon[:-1],self.lat[1:], self.lon[1:])
        self.dist = np.concatenate(([0],dist))*1e-3 # [km]    
        self.distcum = np.cumsum(self.dist)
        return 

    def get_bathy(self,res='hr'):
        if   res=='lr':
            print(' ... get bathymetry from ',self.file_topo_lr,' ...')
            nc     = Dataset(self.file_topo_lr,'r')
            lonh   = nc.variables['x'][3800:5200]
            lath   = nc.variables['y'][4100:4900]
            self.h = nc.variables['z'][4100:4900,3800:5200]
        elif res=='hr':
            print(' ... get bathymetry from ',self.file_topo_hr,' ...')
            nc     = Dataset(self.file_topo_hr,'r')
            lonh   = nc.variables['lon'][:]
            lath   = nc.variables['lat'][:]
            self.h = nc.variables['z'][:]
        nc.close()
        self.lonh,self.lath = np.meshgrid(lonh,lath)
        return

    def interpolate_bathy(self):
        print(' ... interpolate bathymetry on a fine-resolution grid ... ')
        h_nonan = np.copy(self.h.T); h_nonan[np.isnan(h_nonan)]=0 # in srtm15 dataset 
        spline = itp.RectBivariateSpline(self.lonh[0,:],self.lath[:,0],h_nonan,kx=1,ky=1)
        del(h_nonan)
        ddeg   = self.lonh[0,1]-self.lonh[0,0] # resolution of bathy dataset 
        npts   = np.zeros(self.nm-1) # number of points per segment between stations 
        for i in range(self.nm-1):
            npts[i] = np.ceil(np.max((abs(self.lon[i+1]-self.lon[i])/ddeg,
                                      abs(self.lat[i+1]-self.lat[i])/ddeg)))
        lonitp = np.concatenate([np.linspace(self.lon[i],self.lon[i+1],npts[i])
                                for i in range(self.nm-1)])
        latitp = np.concatenate([np.linspace(self.lat[i],self.lat[i+1],npts[i])
                                for i in range(self.nm-1)])
        self.hitp  = spline.ev(lonitp,latitp)
        distitp    = dist_sphere(latitp[:-1],lonitp[:-1],latitp[1:],lonitp[1:])
        distitp    = np.concatenate(([0],distitp))
        self.hdistcum = np.cumsum(distitp)*1e-3 # [km]  
        return

    def interpolate_2d(self,var,dx,method_itp):
        print(' ... interpolate data on an x-z grid ... ')
        # - define the 2-D grid - 
        xe_firstguess = np.arange(self.distcum[0],self.distcum[-1]+dx,dx) # edge
        xe = np.linspace(self.distcum[0],self.distcum[-1],xe_firstguess.shape[0]) # so that it has the right boundaries
        # /!\ consequently, dx slightly differs from the one prescribed ! 
        zc = self.z # uses the native vertical grid 
        dz = zc[1]-zc[0]
        ze = np.concatenate(([zc[0]-0.5*dz],0.5*(zc[1:]+zc[:-1]),[zc[-1]+0.5*dz])) 
        xc = 0.5*(xe[1:]+xe[:-1])
        self.xc_itp,self.zc_itp = np.meshgrid(xc,zc)
        self.xe_itp,self.ze_itp = np.meshgrid(xe,ze)
        dist_tile = np.tile(self.distcum,(self.nz,1)).T
        z_tile    = np.tile(self.z,(self.nm,1))
        # - interpolate data -
        self.nx = xc.shape[0]
        var_itp = np.zeros((self.nt,self.nz,self.nx))
        for t in range(self.nt): 
            tmp = var[:,t,:]
            var_itp[t,:,:] = itp.griddata((np.ravel(dist_tile[~np.isnan(tmp)]),
                                           np.ravel(z_tile[~np.isnan(tmp)])),
                                           np.ravel(tmp[~np.isnan(tmp)]),
                                           (self.xc_itp,self.zc_itp),method=method_itp)
        return var_itp

    def get_date(self): 
        print(' ... get the date ... ')
        # reference times are different in data and Python
        diff_days     = (datetime.datetime(1950,1,1) - datetime.datetime(1970,1,1)).days 
        self.time_fmt = [tm.gmtime((i+diff_days)*86400) for i in self.time]
        self.ymd      = [datetime.date(self.time_fmt[i][0],self.time_fmt[i][1],self.time_fmt[i][2]) 
                         for i in np.arange(self.nt)]
        self.ymdhms   = [datetime.datetime(self.time_fmt[i][0],self.time_fmt[i][1],self.time_fmt[i][2],
                                     self.time_fmt[i][3],self.time_fmt[i][4],self.time_fmt[i][5]) 
                         for i in np.arange(self.nt)]
        # --- correction to round to minutes --- 
        for t in range(self.nt): 
            self.ymdhms[t] = round_time(self.ymdhms[t],round_to=60)
        # --- get day of year - round number ---
        self.doy = [self.ymdhms[i].timetuple().tm_yday for i in np.arange(self.nt)]
        # --- get day after mooring deployment - decimal number --- 
        self.dayad = self.time - self.time[0]  
        return

    def get_MLD_from_WOA(self):  
        print(' ... get mixed-layer depth from monthly WOA data ... ') 
        # /!\ ACHTUNG: needs to call get_N2_from_WOA before!  
        # NB: MLD is computed from sigma0 averaged in a box encompassing mooring array  
        # - criterion: density jump of 0.03 kg m-3 from 10-m density (de Boyer Montegut et al 2004) - 
        self.mld = np.zeros(self.nt)  
        for t in range(self.nt):
            sig0 = self.sig0_woa_hourly[t,:] 
            zitp = itp.interp1d(sig0,self.zc_woa) 
            self.mld[t] = zitp(sig0[2]+0.03) # index 2 corresponds to 10 m depth 
        return  

    def get_alpha(self): 
        print(' ... get slope of internal tide beams ... ')
        # /!\ ACHTUNG: needs to call get_N2_from_WOA before!  
        # - vertically integrate N2 -   
        dz    = abs(np.diff(self.zc_woa)) 
        alpha = np.zeros((self.nm,self.nt)) 
        for m in range(self.nm):
            N2bar      = np.nansum(self.N2_woa_hourly*dz,axis=1)/self.seafloor[m]
            fi         = gsw.f(self.lat[m])#/(2*np.pi) 
            M2rad      = M2*2*np.pi # [rad s-1]
            alpha[m,:] = ((M2rad**2-fi**2)/(N2bar-M2rad**2))**0.5  
        self.alpha = alpha
        return 

    def get_N2_from_WOA(self): 
        print(' ... get stratification from monthly WOA data ... ') 
        file_woat = ['/Volumes/cvic_usb/Data/WOA/woa18_decav_t%.2i_01.nc'%i for i in range(13)] #yearly+monthly 
        file_woas = ['/Volumes/cvic_usb/Data/WOA/woa18_decav_s%.2i_01.nc'%i for i in range(13)]
        # --- read yearly data and bathymetry --- 
        print('      --> read data ') 
        nc      = Dataset(file_woat[0],'r')
        lat_woa = nc.variables['lat'][:]
        lon_woa = nc.variables['lon'][:]
        imin    = np.nanargmin(abs(lon_woa-(-34))) # mean profile over a box encompassing mooring array
        imax    = np.nanargmin(abs(lon_woa-(-28))) 
        jmin    = np.nanargmin(abs(lat_woa-57)) 
        jmax    = np.nanargmin(abs(lat_woa-60))
        lon_avg = np.nanmean(lon_woa[imin:imax+1])
        lat_avg = np.nanmean(lat_woa[jmin:jmax+1])
        zc      = -nc.variables['depth'][:]
        kmin    = np.argwhere(zc==-1500)[0][0]+1
        t_yr    = np.nanmean(nc.variables['t_an'][0,kmin:,jmin:jmax+1,imin:imax+1],axis=(-1,-2))
        nc.close()
        nc      = Dataset(file_woas[0],'r')
        s_yr    = np.nanmean(nc.variables['s_an'][0,kmin:,jmin:jmax+1,imin:imax+1],axis=(-1,-2))
        nc.close()
        # --- read monthly data --- 
        nz   = zc.shape[0]-1 
        N2   = np.zeros((12,nz))
        sig0 = np.zeros((12,nz+1))
        dtdz = np.zeros((12,nz)) # vertical gradient of temperature 
        for m in range(12):
            nc = Dataset(file_woat[m+1],'r')
            t  = np.squeeze(np.nanmean(nc.variables['t_an'][:,:,jmin:jmax+1,imin:imax+1],axis=(-1,-2)))
            nc.close()
            t  = np.concatenate((t,t_yr),axis=0)
            nc = Dataset(file_woas[m+1],'r')
            s  = np.squeeze(np.nanmean(nc.variables['s_an'][:,:,jmin:jmax+1,imin:imax+1],axis=(-1,-2)))
            nc.close()
            s  = np.concatenate((s,s_yr),axis=0)
            # - following instructions in MacDougall and Barker 2011 - 
            p  = gsw.p_from_z(zc,lat_avg)
            SA = gsw.SA_from_SP(s,p,lon_avg,lat_avg)
            CT = gsw.CT_from_t(SA,t,p)
            [N2month,p_mid] = gsw.Nsquared(SA,CT,p,lat_avg) # [(rad s^-1)^2] 
            ze   = gsw.z_from_p(p_mid.data,lat_avg)
            N2month[N2month<0] = 1e-8 # smallest stratification at very deep seafloors 
            N2[m,:] = N2month
            sig0[m,:] = gsw.sigma0(SA,CT)
            dtdz[m,:] = np.diff(CT)/np.diff(zc) 
        # --- a bit of smoothing --- 
        ns = 4  # running-mean window half-length 
        N2_woa_monthly   = np.copy(N2) 
        sig0_woa_monthly = np.copy(sig0) # no smoothing on density  
        dtdz_woa_monthly = np.copy(dtdz) 
        for k in range(ns,nz-ns): 
            N2_woa_monthly[:,k] = np.nanmean(N2[:,k-ns:k+ns+1],axis=1)  
            dtdz_woa_monthly[:,k] = np.nanmean(dtdz[:,k-ns:k+ns+1],axis=1)  
        for k in range(ns): # top-most values 
            N2_woa_monthly[:,k] = np.nanmean(N2[:,:k+ns+1],axis=1) 
            dtdz_woa_monthly[:,k] = np.nanmean(dtdz[:,:k+ns+1],axis=1) 
        for k in range(nz-ns,nz): # bottom-most values 
            N2_woa_monthly[:,k] = np.nanmean(N2[:,k:],axis=1)
            dtdz_woa_monthly[:,k] = np.nanmean(dtdz[:,k:],axis=1)
        if 0: 
            plt.figure() 
            cmap = plt.get_cmap('gnuplot') 
            colors = [cmap(i) for i in np.linspace(0, 1, 12)] 
            for m in range(12): 
                plt.semilogx(N2_woa_monthly[m,:],1e-3*ze,color=colors[m],lw=0.3,label='%.2i'%(m+1)) 
                plt.legend()
                plt.ylabel('Depth [km]')
                plt.xlabel('$N^2$ [(rad s$^{-1}$)$^2$]') 
                plt.ylim(-2.5,0)
            plt.savefig('../Figures/moorings_N2_monthly_profiles_WOA.pdf',bbox_inches='tight')  
        # - output variables - 
        self.N2_woa_monthly   = N2_woa_monthly
        self.dtdz_woa_monthly = dtdz_woa_monthly
        self.zc_woa = zc 
        self.ze_woa = ze 
        
        # ------ interpolate at data time ------
        print('      --> interpolate in time ') 
        doy_woa = [datetime.datetime(2015,i,15).timetuple().tm_yday for i in range(1,13)]
        # - padding -   
        doy_woa.insert(0,doy_woa[0]-30) 
        doy_woa.insert(len(doy_woa),doy_woa[-1]+30)
        N2_woa_monthly   = np.concatenate(([N2_woa_monthly[-1,:]],N2_woa_monthly,[N2_woa_monthly[0,:]]))
        sig0_woa_monthly = np.concatenate(([sig0_woa_monthly[-1,:]],sig0_woa_monthly,[sig0_woa_monthly[0,:]]))
        dtdz_woa_monthly = np.concatenate(([dtdz_woa_monthly[-1,:]],dtdz_woa_monthly,[dtdz_woa_monthly[0,:]]))
        nz_woa          = ze.shape[0]
        N2_woa_hourly   = np.zeros((self.nt,nz_woa)) 
        sig0_woa_hourly = np.zeros((self.nt,nz_woa+1)) 
        dtdz_woa_hourly = np.zeros((self.nt,nz_woa)) 
        for k in range(nz_woa): 
            itp_t                = itp.interp1d(doy_woa,N2_woa_monthly[:,k])        
            N2_woa_hourly[:,k]   = itp_t(self.doy) 
            itp_t                = itp.interp1d(doy_woa,dtdz_woa_monthly[:,k])        
            dtdz_woa_hourly[:,k] = itp_t(self.doy) 
        for k in range(nz_woa+1): 
            itp_t                = itp.interp1d(doy_woa,sig0_woa_monthly[:,k])        
            sig0_woa_hourly[:,k] = itp_t(self.doy) 
        # - output variables - 
        self.N2_woa_hourly   = N2_woa_hourly  
        self.sig0_woa_hourly = sig0_woa_hourly  
        
        # ------ interpolate on the vertical ------  
        print('      --> interpolate on the vertical ') 
        if self.level=='L4': # ------> interpolate on data regular grid  
            N2_woa_grid   = np.zeros((self.nt,self.nz))
            sig0_woa_grid = np.zeros((self.nt,self.nz))   
            dtdz_woa_grid = np.zeros((self.nt,self.nz))   
            for t in range(self.nt): 
                itp_z              = itp.interp1d(ze,N2_woa_hourly[t,:],bounds_error=False) 
                N2_woa_grid[t,:]   = itp_z(self.zc.data)  
                itp_z              = itp.interp1d(zc,sig0_woa_hourly[t,:],bounds_error=False) 
                sig0_woa_grid[t,:] = itp_z(self.zc.data)  
                itp_z              = itp.interp1d(ze,dtdz_woa_hourly[t,:],bounds_error=False) 
                dtdz_woa_grid[t,:] = itp_z(self.zc.data)  
            # - output variables - 
            self.N2_woa_grid   = N2_woa_grid 
            self.sig0_woa_grid = sig0_woa_grid  
            self.dtdz_woa_grid = dtdz_woa_grid  
        elif self.level=='L3': # ------> interpolate at data *mean* depth (easier)  
            # - at T,S points -  
            N2_zts   = np.zeros((self.nt,self.nz_ts))
            sig0_zts = np.zeros((self.nt,self.nz_ts))   
            dtdz_zts = np.zeros((self.nt,self.nz_ts))   
            zavg     = np.nanmean(self.z_ts.data,axis=0)
            for t in range(self.nt): 
                itp_z         = itp.interp1d(ze,N2_woa_hourly[t,:]) 
                N2_zts[t,:]   = itp_z(zavg)  
                itp_z         = itp.interp1d(zc,sig0_woa_hourly[t,:]) 
                sig0_zts[t,:] = itp_z(zavg)  
                itp_z         = itp.interp1d(ze,dtdz_woa_hourly[t,:]) 
                dtdz_zts[t,:] = itp_z(zavg)  
            # - at u,v points -  
            N2_zuv = np.zeros((self.nt,self.nz_uv))
            zavg   = np.nanmean(self.z_uv.data,axis=0)
            for t in range(self.nt): 
                itp_z         = itp.interp1d(ze,N2_woa_hourly[t,:]) 
                N2_zuv[t,:]   = itp_z(zavg)  
            # - output variables - 
            self.N2_woa_zts   = N2_zts
            self.sig0_woa_zts = sig0_zts
            self.dtdz_woa_zts = dtdz_zts
            self.N2_woa_zuv   = N2_zuv
        return



##################################################################################
#################################### OLD STUFF ###################################
##################################################################################
''' 
    def time_filtering(self,varname='u',filter_type='sd'): 
        print(' ... time filtering of the data ... ') 
        if filter_type not in ['sd','ni','si','lp']: 
            exit('ERROR: filter_type must be in [sd,ni,si,lp], i.e., \n'\
                 '       semi-diurnal,near-inertial,sub-inertial,low-pass')
        freq_longname={'sd':'semi-diurnal','ni':'near-inertial','si':'sub-inertial','lp':'low-pass'}
        print('     --> filtering variable ',varname,' at ',freq_longname[filter_type],' frequency')
        # --- set up filtering parameters --- 
        c           = 1.07 # Alford's parameter is 1.25 
        Nbut        = 4 # order of Butterworth filter
        dt          = round((self.time[1]-self.time[0])*24*3600) # [sec] 
        fi          = gsw.f(np.mean(self.lat))/(2*np.pi)
        M2          = 1./44700. # [sec-1] 
        S2          = 1./43200.
        sd          = 0.5*(M2+S2) 
        if filter_type=='sd': # semi-diurnal frequencies
            Wn  = np.array([(1./c)*sd,c*sd])*(2*dt)
            b,a = sig.butter(Nbut,Wn,btype='bandpass',output='ba')  
        elif filter_type=='ni': # inertial frequency  
            Wn  = np.array([(1./c)*fi,c*fi])*(2*dt)
            b,a = sig.butter(Nbut,Wn,btype='bandpass',output='ba')  
        elif filter_type=='si': # subinertial frequency
            #Wn = 0.7*fi*(2*dt) 
            Wn = (1./c)*fi*(2*dt) 
            b,a = sig.butter(Nbut,Wn,btype='lowpass',output='ba')  
        elif filter_type=='lp': # low-pass at 0.1*f = mesoscale 
            Wn = 0.1*fi*(2*dt) 
            b,a = sig.butter(Nbut,Wn,btype='lowpass',output='ba')  
        # --- choose variables --- 
        if   varname=='u': var = self.u  
        elif varname=='v': var = self.v  
        elif varname=='t': var = self.t  
        elif varname=='s': var = self.s  
        # --- loop on moorings and depth to better handle nans --- 
        if self.nm==1: var=np.tile(var,(1,1,1)) # duplicate to avoid if statements in the loop  
        var_filt = np.copy(var)*np.nan  
        for m in range(self.nm):
            print('         * mooring %i'%m)  
            for k in range(var.shape[-1]):  
                tmp = np.copy(var[m,:,k])  
                tmin = 0 
                if tmp[~np.isnan(tmp)].shape[0]>=10: # filter only if there is at least 10 non-nan values 
                    while tmin<self.nt-2:  
                        # - get intervals of non-nan data - 
                        while np.isnan(tmp[tmin]) and tmin<self.nt-2:tmin+=1
                        tmax = tmin+1  
                        while ~np.isnan(tmp[tmax]) and tmax<self.nt-1:tmax+=1 
                        if tmax-tmin>27: # ValueError: The length of the input vector x must be at least padlen, which is 27. 
                            if filter_type in ['sd','ni']:
                                var_filt[m,tmin:tmax,k] = sig.filtfilt(b,a,tmp[tmin:tmax]) 
                            elif filter_type in ['si','lp']:
                                var_filt[m,tmin:tmax,k] = sig.filtfilt(b,a,tmp[tmin:tmax],method='gust') 
                        tmin = tmax+1
        # --- return filtered variable --- 
        if varname=='u' and filter_type=='sd' and self.nm>1:  self.u_sd = var_filt
        if varname=='u' and filter_type=='sd' and self.nm==1: self.u_sd = var_filt[0,:,:] 
        if varname=='u' and filter_type=='ni' and self.nm>1:  self.u_ni = var_filt
        if varname=='u' and filter_type=='ni' and self.nm==1: self.u_ni = var_filt[0,:,:] 
        if varname=='u' and filter_type=='si' and self.nm>1:  self.u_si = var_filt
        if varname=='u' and filter_type=='si' and self.nm==1: self.u_si = var_filt[0,:,:] 
        if varname=='u' and filter_type=='lp' and self.nm>1:  self.u_lp = var_filt
        if varname=='u' and filter_type=='lp' and self.nm==1: self.u_lp = var_filt[0,:,:] 
        if varname=='v' and filter_type=='sd' and self.nm>1:  self.v_sd = var_filt
        if varname=='v' and filter_type=='sd' and self.nm==1: self.v_sd = var_filt[0,:,:] 
        if varname=='v' and filter_type=='ni' and self.nm>1:  self.v_ni = var_filt
        if varname=='v' and filter_type=='ni' and self.nm==1: self.v_ni = var_filt[0,:,:] 
        if varname=='v' and filter_type=='si' and self.nm>1:  self.v_si = var_filt
        if varname=='v' and filter_type=='si' and self.nm==1: self.v_si = var_filt[0,:,:] 
        if varname=='v' and filter_type=='lp' and self.nm>1:  self.v_lp = var_filt
        if varname=='v' and filter_type=='lp' and self.nm==1: self.v_lp = var_filt[0,:,:] 
        if varname=='t' and filter_type=='sd' and self.nm>1:  self.t_sd = var_filt
        if varname=='t' and filter_type=='sd' and self.nm==1: self.t_sd = var_filt[0,:,:] 
        if varname=='t' and filter_type=='ni' and self.nm>1:  self.t_ni = var_filt
        if varname=='t' and filter_type=='ni' and self.nm==1: self.t_ni = var_filt[0,:,:] 
        if varname=='t' and filter_type=='si' and self.nm>1:  self.t_si = var_filt
        if varname=='t' and filter_type=='si' and self.nm==1: self.t_si = var_filt[0,:,:] 
        if varname=='t' and filter_type=='lp' and self.nm>1:  self.t_lp = var_filt
        if varname=='t' and filter_type=='lp' and self.nm==1: self.t_lp = var_filt[0,:,:] 
        return
'''    
