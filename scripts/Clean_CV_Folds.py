#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:40:40 2019

@author: deeperthought
"""

import pandas as pd
import os

#PATH = '/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/kirby_data/'
PATH = '/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/Liz_Mary/With_1st_batch/'


folds = [PATH + x for x in os.listdir(PATH)]

TRAIN = [x for x in folds if 'train' in x]
VAL = [x for x in folds if 'val' in x]
TEST = [x for x in folds if 'test' in x]


#BLACKLIST = [x for x in folds if 'Blacklist' in x][0]
BLACKLIST = '/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/BLACKLIST_Old.txt'


BLACKLIST = pd.read_csv(BLACKLIST, header=None)

BLACKLIST[0] = BLACKLIST[0].apply(lambda x : '/'.join(x.split('/')[-2:-1]))

T1POST = [x for x in folds if 't1post' in x]
T1POST_TRAIN = [x for x in T1POST if 'train' in x]
T1POST_VAL = [x for x in T1POST if 'val' in x]
T1POST_TEST = [x for x in T1POST if 'test' in x]

index_to_remove_train = []
index_to_remove_test = []
index_to_remove_val = []


check_correct_pairings = {}


for l in TRAIN:
    scans = pd.read_csv(l, header=None)
#    scans[0] = scans[0].apply(lambda x : '/'.join(x.split('/')[-2:-1]))
    scans[0] = scans[0].apply(lambda x : 'MSKCC_' + x.split('MSKCC_')[-1][:23])
    check_correct_pairings[l.split('/')[-1].replace('.txt','')] = scans[0]
    index_to_remove_train.append(scans.loc[scans[0].isin(list(BLACKLIST[0]))].index)

df = pd.DataFrame(check_correct_pairings)
df['Malignants_check'] = df.apply(lambda x : x['train_labels']==x['train_sub']==x['train_t1post']==x['train_t2'], axis=1)

df.loc[df['train_labels'].str.contains('_16-328')]
df['Malignants_check'].value_counts()

df['MRI_check'] = df.apply(lambda x : x['train_sub']==x['train_t1post']==x['train_t2'], axis=1)
df['MRI_check'].value_counts()

df.columns

i = 0
for l in TRAIN:
    print(l)
    scans = pd.read_csv(l, header=None)
    scans = scans.drop(index=index_to_remove_train[i]) 
    scans.to_csv(l, index=False, header=False)
    i += 1

for l in VAL:
    scans = pd.read_csv(l, header=None)
    scans[0] = scans[0].apply(lambda x : 'MSKCC_' + x.split('MSKCC_')[-1][:23])
    index_to_remove_val.append(scans.loc[scans[0].isin(list(BLACKLIST[0]))].index)
    
i = 0    
for l in VAL:
    print(l)
    scans = pd.read_csv(l, header=None)
    scans = scans.drop(index=index_to_remove_val[i]) 
    scans.to_csv(l, index=False, header=False)    
    
    
for l in TEST:
    scans = pd.read_csv(l, header=None)
    scans[0] = scans[0].apply(lambda x : 'MSKCC_' + x.split('MSKCC_')[-1][:23])
    index_to_remove_test.append(scans.loc[scans[0].isin(list(BLACKLIST[0]))].index)    
    
i = 0    
for l in TEST:
    print(l)
    scans = pd.read_csv(l, header=None)
    scans = scans.drop(index=index_to_remove_test[i]) 
    scans.to_csv(l, index=False, header=False)    