import numpy as np
import matplotlib.pyplot as plt

adcp = np.load('ADCP_uv_ni_wkb.npz')
KE_ni = adcp['KE_ni_wkb']

plt.plot(KE_ni[:, 0])
plt.show()
