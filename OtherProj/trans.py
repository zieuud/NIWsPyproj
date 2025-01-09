import numpy as np
from scipy.io import savemat

caseName = ['2_3', '2_4', '3_3', '3_4', '4_3', '4_4']
# caseName = ['2']
for case in caseName:

    data = np.load('mode_decom_{}.npz'.format(case))
    savemat(r'L:\NIWs\ExtractData\CaseV5_{}\mode_decoms.mat'.format(case), data)
    # savemat(r'L:\NIWs\ExtractData\CaseV5\mode_decoms.mat', data)

    data = np.load('ParallelLine4_uv_filter_mod_{}.npz'.format(case))
    savemat(r'L:\NIWs\ExtractData\CaseV5_{}\ParallelLine4_uv_filter_mod.mat'.format(case), data)
    # savemat(r'L:\NIWs\ExtractData\CaseV5\ParallelLine4_uv_filter_mod.mat', data)
