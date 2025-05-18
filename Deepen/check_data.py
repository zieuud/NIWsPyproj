import netCDF4 as nc


data1 = nc.Dataset(r'L:\graduation proj\data\MOOR\RidgeMix_1mooring\Mooring_data\JR15007_Moorings_ADCP_merged.nc')
data2 = nc.Dataset(r'L:\graduation proj\data\MOOR\RidgeMix_1mooring\Mooring_data\JR15007_Moorings_all_sensors.nc')
print(data2.variables)