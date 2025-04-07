import netCDF4 as nc
from datetime import datetime, timedelta

data = nc.Dataset('L:\graduation proj\data\MOOR\RidgeMix_1mooring\Mooring_data\JR15007_Moorings_ADCP_merged.nc')
time = data.variables['time'][:]

date = [datetime(1970, 1, 1) + timedelta(days=float(i)) for i in time]
print('c')
