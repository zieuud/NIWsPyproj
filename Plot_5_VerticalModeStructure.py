import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker


data1 = np.load(r'ReanaData/WOA23_pmodes_moorGrid_yearly.npz')
pmodes = data1['pmodes']
Nsq = data1['Nsq']
ze = data1['ze']
data2 = np.load(r'MoorData/ADCP_uv.npz')
depth = data2['depth_adcp']

fig = plt.figure(1, figsize=(10, 12))
gs = gridspec.GridSpec(2, 6, hspace=0, wspace=0)
for i in range(11):
    if i >= 6:
        j = i + 1
    else:
        j = i
    if i == 0:
        ax1 = fig.add_subplot(gs[j])
        ax1.plot(Nsq, ze, 'k-')
        ax1.set_ylabel('Depth (m)')
        ax1.set_title(r'$N^{2}$ $(rad/s)^{2}$')
        ax1.set_yticks([0, -500, -1000, -1500, -2000])
        ax1.ticklabel_format(style='sci', scilimits=(0, 10e-5), axis='x')
    else:
        ax = fig.add_subplot(gs[j])
        if pmodes[i, 0] < 0:
            pmodes[i, :] = -pmodes[i, :]
        ax.plot(pmodes[i, :].T, depth, 'k-')
        if i == 6:
            ax.set_ylabel('Depth (m)')
            ax.set_yticks([-500, -1000, -1500, -2000])
        else:
            ax.set_yticks([])
        ax.set_xlim(-np.nanmax(np.abs(pmodes[i, :])) - 0.1, np.nanmax(np.abs(pmodes[i, :])) + 0.1)
        ax.set_xticks([])
        ax.plot([0, 0], [0, -2000], 'k--')
        ax.text(0, -1800, 'mode {}'.format(i), fontsize=16, va='center', ha='center')
        ax.set_ylim(-2000, 0)

# plt.savefig(r'figures/fig_5_VerticalModeStructure.jpg', dpi=300, bbox_inches='tight')
# plt.savefig(r'figuresFinal/fig_5_VerticalModeStructure.png', dpi=300, bbox_inches='tight')
# plt.savefig(r'figuresFinal/fig_5_VerticalModeStructure.pdf', bbox_inches='tight')
plt.show()
