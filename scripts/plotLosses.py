#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:29:04 2019

@author: deeperthought
"""

import numpy as np
import matplotlib.pyplot as plt

SAVE_FIG = False
SMOOTH = 50
ITERATIONS = 2500

PATH = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/'

# TOY DATA
big = PATH + 'MultiPriors_v2_MSKCC_configFile_MultiPriors_v2_F4_Overfit_test_2019-12-02_1437/LOSS.npy'
big2 = PATH + 'MultiPriors_v2_MSKCC_configFile_MultiPriors_v2_F4_Overfit_test_2019-12-02_1601/LOSS.npy'
small = PATH + 'MultiPriors_v2_MSKCC_configFile_MultiPriors_v2_F4_Overfit_test_2019-12-02_1526/LOSS.npy'
old_small = PATH + 'MultiPriors_v2_MSKCC_configFile_MultiPriors_v2_F4_parallel_classBalanced_2019-11-23_1839 (copy)/LOSS.npy'
tiny = PATH + 'MultiPriors_v2_MSKCC_configFile_MultiPriors_v2_F4_Overfit_test_2019-12-02_1612/LOSS.npy'
tiniest = PATH + 'MultiPriors_v2_MSKCC_configFile_MultiPriors_v2_F4_Overfit_test_2019-12-02_1629/LOSS.npy'
bigModel_1M_noL2 = PATH + 'MultiPriors_v2_Big_MSKCC_configFile_MultiPriors_v2_BIG_F4_overfit_test_2019-12-03_1046/LOSS.npy'
bigModel_3M_noL2 = PATH + 'MultiPriors_v2_Big_MSKCC_configFile_MultiPriors_v2_BIG_F4_overfit_test_2019-12-03_1150/LOSS.npy'

# REAL DATA
tiny_model_realData = PATH + 'MultiPriors_v2_MSKCC_configFile_MultiPriors_v2_F4_parallel_classBalanced_2019-12-02_1741/LOSS.npy'
big_realData = PATH + 'MultiPriors_v2_Big_MSKCC_configFile_MultiPriors_v2_BIG_F4_parallel_classBalanced_2019-11-30_2030/LOSS.npy'
normal_realData = PATH + 'MultiPriors_v2_MSKCC_configFile_MultiPriors_v2_F4_parallel_classBalanced_2019-11-23_1839 (copy)/LOSS.npy'
bigModels_small_realData = PATH + 'MultiPriors_v2_Big_MSKCC_configFile_MultiPriors_v2_BIG_F4_parallel_classBalanced_2019-12-03_1015/LOSS.npy'
bigModel_1M_realData = PATH + 'MultiPriors_v2_Big_MSKCC_configFile_MultiPriors_v2_BIG_F4_parallel_classBalanced_2019-12-03_1221/LOSS.npy'
bigModel_1M_realData2 = PATH + 'MultiPriors_v2_Big_MSKCC_configFile_MultiPriors_v2_BIG_F4_parallel_classBalanced_2019-12-03_1829/LOSS.npy'
bigModel_09M_frozenbreastMask_realData = PATH + 'MultiPriors_v2_Big_BreastMask_MSKCC_configFile_MultiPriors_v2_BIG_BreastMask_F4__2019-12-05_1113/LOSS.npy'
bigModel_frozenBreastMask_concatChannels = PATH + 'MultiPriors_v2_Big_BreastMask_MSKCC_configFile_MultiPriors_v2_BIG_BreastMask_F4__2019-12-05_1531/LOSS.npy'

# SELECTION
#models = [bigModel_3M_noL2, big, big2, bigModel_1M_noL2, small, old_small, tiny, tiniest]
#legend = ['3M_noL2','2.7M','2.4M', '1M_noL2','0.7M','0.5M', '15K', '3K', '15K_RealData']

#models = [tiny_model_realData, normal_realData, big_realData, bigModels_small_realData, bigModel_1M_realData2, bigModel_09M_frozenbreastMask_realData]
#legend = ['smallModel_15K','smallModel_700K', 'bigModel_1M', 'bigModel_27K', 'bigModel_1M_noL2', 'bigModel_0.9M_frozenBM_noL2']

models = [ bigModel_1M_realData2, bigModel_09M_frozenbreastMask_realData, bigModel_frozenBreastMask_concatChannels]
legend = ['bigModel_1M_noL2', 'bigModel_0.9M_frozenBM_noL2', 'bigModel_0.9M_frozenBM_concatChannels_noL2']

assert len(models) == len(legend)
#%%
def movingAverageConv(a, window_size=1) :
    if window_size == 1:
        return a
    #if not a : return a
    window = np.ones(int(window_size))
    result = np.convolve(a, window, 'full')[ : len(a)] # Convolve full returns array of shape ( M + N - 1 ).
    slotsWithIncompleteConvolution = min(len(a), window_size-1)
    result[slotsWithIncompleteConvolution:] = result[slotsWithIncompleteConvolution:]/float(window_size)
    if slotsWithIncompleteConvolution > 1 :
        divisorArr = np.asarray(range(1, slotsWithIncompleteConvolution+1, 1), dtype=float)
        result[ : slotsWithIncompleteConvolution] = result[ : slotsWithIncompleteConvolution] / divisorArr
    return result



myArrays = []
for arr in models:
    loaded = np.load(arr)
    smooth_arr = movingAverageConv(loaded, SMOOTH)
    myArrays.append(smooth_arr)

for arr in myArrays:
    plt.plot(arr[:ITERATIONS])
plt.legend(legend)
plt.ylabel('Dice Loss on Training Set')
plt.xlabel('Training iteration')
plt.ylim([0,0.8])
plt.grid(b=True, which='major',zorder=1)
plt.minorticks_on()
plt.grid(b=True, which='minor',alpha=0.35, zorder=1) 

if SAVE_FIG:
    plt.savefig(PATH + 'Model_Size_Loss.png', dpi=200)