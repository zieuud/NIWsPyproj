import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def find_date(date, year, month, day):
    for i in range(len(date)):
        if date[i].year == year and date[i].month == month and date[i].day == day:
            print(i)
            return i


def find_depth(depth, d):
    idx = np.argmin(abs(depth + d))
    print(idx)
    return idx


N2 = np.load(r'ReanaData\WOA23_N2_grid.npy')
adcp = np.load('ADCP_uv_ni_wkb.npz')
KE_ni = adcp['KE_ni_wkb']
adcp0 = np.load('ADCP_uv.npz')
moorDate = adcp0['mtime']
dateForPlot = [datetime(1, 1, 1) + timedelta(days=m-366) for m in moorDate]
depth = adcp0['depth']
woa23Temp = np.load(r'ReanaData\WOA23_temp_grid.npy')
# ---------- find the depth of mixed layer ----------
ml_depthArray = []
for idx in range(np.size(woa23Temp, 0)):
    ml_idx = np.argwhere(woa23Temp[idx, :] - woa23Temp[idx, 0] < -0.5)[0]
    ml_depthArray.append(depth[ml_idx])
ml_depthArray = np.array(ml_depthArray)

find_date(dateForPlot, 2016, 3, 16)
find_date(dateForPlot, 2016, 3, 23)
find_depth(depth, 200)
find_depth(depth, 1400)
gs1 = abs(depth[27] - depth[0]) / (dateForPlot[595] - dateForPlot[403]).days
gs2 = abs(depth[30] - depth[0]) / (dateForPlot[2080] - dateForPlot[1987]).days
gs3 = abs(depth[171] - depth[153]) / (dateForPlot[1747] - dateForPlot[1603]).days
gs4 = abs(depth[21] - depth[0]) / (dateForPlot[4243] - dateForPlot[4075]).days
# line1: date: 403 595 depth: 0 27
# line2: date: 1987 2080 depth: 0 30
# line3: date: 1603 1747 depth: 153 171
# line4: date: 4123 4411 depth: 0 21
# --------- plot KE_ni_wkb profile ----------
plt.figure(1, figsize=(10, 6))
[depth_mesh, moorDate_mesh] = np.meshgrid(depth[:], dateForPlot)
c = plt.pcolor(moorDate_mesh, depth_mesh, KE_ni[:, :], cmap='Oranges', vmin=0, vmax=10)
cb = plt.colorbar(c)
plt.plot([dateForPlot[403], dateForPlot[595]], [depth[0], depth[27]], 'b-')
plt.text(dateForPlot[595], depth[27], r'cgz={}'.format(gs1))
plt.plot([dateForPlot[1987], dateForPlot[2080]], [depth[0], depth[30]], 'b-')
plt.text(dateForPlot[2080], depth[30], r'cgz={}'.format(gs2))
plt.plot([dateForPlot[1603], dateForPlot[1747]], [depth[153], depth[171]], 'b-')
plt.text(dateForPlot[1747], depth[171], r'cgz={}'.format(gs3))
plt.plot([dateForPlot[4075], dateForPlot[4243]], [depth[0], depth[19]], 'b-')
plt.text(dateForPlot[4243], depth[19], r'cgz={}'.format(gs4))
cb.set_label(r'$KE_{NI}^{WKB}$ $(J/m^{3})$')
plt.ylabel('depth (m)')
line, = plt.plot(dateForPlot, ml_depthArray, label='$H_{ML}$')
plt.legend(handles=[line])
plt.savefig(r'figures\check.jpg', dpi=300)
# plt.show()
print('c')