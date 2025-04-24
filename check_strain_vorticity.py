import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from scipy import interpolate


# use GLORYS data
vorticity_moor_hourly1 = np.load(r'ReanaData\GLORYS_vorticity.npy')[:, :180]
strain_moor_hourly1 = np.load(r'ReanaData\GLORYS_strain.npy')[:, :180]
# use AVISO data
vorticity_moor_hourly2 = np.load(r'ReanaData\AVISO_0125_vorticity3.npy')
vorticity_moor_hourly2 = np.tile(vorticity_moor_hourly2, (180, 1)).T
strain_moor_hourly2 = np.load(r'ReanaData\AVISO_0125_strain4.npy')
strain_moor_hourly2 = np.tile(strain_moor_hourly2, (180, 1)).T

lat_moor = 36.23
fi = 2 * 7.292e-5 * np.sin(lat_moor/180*np.pi)

vf1 = vorticity_moor_hourly1 / fi
sf1 = strain_moor_hourly1 / fi

vf2 = vorticity_moor_hourly2 / fi
sf2 = strain_moor_hourly2 / fi

plt.figure(1)
plt.plot(vf1[:, 0])
plt.plot(sf1[:, 0])
plt.legend(['vf1', 'sf1'])

plt.figure(2)
plt.plot(vf2[:, 0])
plt.plot(sf2[:, 0])
plt.legend(['vf2', 'sf2'])
plt.show()

