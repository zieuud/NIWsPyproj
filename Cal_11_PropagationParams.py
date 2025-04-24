import numpy as np


def cal_prop_params(cpz_coord, cgz_coord):
    lat_moor = 36.23
    fi = 2 * 7.292e-5 * np.sin(np.deg2rad(lat_moor))
    N0 = 3.6e-3
    cpz5 = []
    cgz5 = []
    m5 = []
    omega5 = []
    kh5 = []
    for i in range(5):
        xp1, yp1, xp2, yp2 = cpz_coord[i]
        xg1, yg1, xg2, yg2 = cgz_coord[i]
        cpz = (yp2 - yp1) / (xp2 - xp1) / 3600
        cgz = (yg2 - yg1) / (xg2 - xg1) / 3600
        m = cal_square_equation(cpz ** 2, -2 * cgz * fi, -fi ** 2)[-1]
        omega = cal_square_equation(1, -cgz * m, -fi ** 2)[0]
        kh = np.sqrt((omega ** 2 - fi ** 2) * m ** 2 / N0 ** 2)
        cpz5.append(cpz)
        cgz5.append(cgz)
        m5.append(m)
        omega5.append(omega / fi)
        kh5.append(kh)
    return np.array(cpz5), np.array(cgz5), np.array(m5), np.array(omega5), np.array(kh5)


def cal_square_equation(a, b, c):
    return (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a), (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)


cpz_coord = [[573, -93, 563, -247], [2047, -138, 2038, -300], [2936, -83, 2922, -222],
             [4219, -29, 4200, -325], [4886, -29, 4879, -160]]
cgz_coord = [[360, -35, 552, -176], [2068, -63, 2285, -179], [2866, -59, 2932, -127],
             [4119, -32, 4269, -171], [4818, -33, 4887, -80]]
lat_moor = 36.23
fi = 2 * 7.292e-5 * np.sin(np.deg2rad(lat_moor))
cpz5, cgz5, m5, omega5, kh5 = cal_prop_params(cpz_coord, cgz_coord)

cpz = ['{:.0f}'.format(i) for i in cpz5*86400]
cpz.append('{:.0f}±{:.0f}'.format(np.mean(cpz5*86400), np.std(cpz5*86400)))
cgz = ['{:.0f}'.format(-i) for i in cgz5*86400]
cgz.append('{:.0f}±{:.0f}'.format(np.mean(-cgz5*86400), np.std(-cgz5*86400)))
omega = ['{:.4}f'.format(i) for i in omega5]
omega.append('{:.4}f±{:.2}f'.format(np.mean(omega5), np.std(omega5)))
verticalWavelength = ['{:.0f}'.format(-2*np.pi/i) for i in m5]
verticalWavelength.append('{:.0f}±{:.0f}'.format(np.mean(-2*np.pi/m5), np.std(-2*np.pi/m5)))
horizontalWaveLength = ['{:.0f}'.format(2*np.pi/i/1000) for i in kh5]
horizontalWaveLength.append('{:.0f}±{:.0f}'.format(np.mean(-2*np.pi/kh5/1000), np.std(-2*np.pi/kh5/1000)))

print(cpz)
print(cgz)
print(omega)
print(verticalWavelength)
print(horizontalWaveLength)