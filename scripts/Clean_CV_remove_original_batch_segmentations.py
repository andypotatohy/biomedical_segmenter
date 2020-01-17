#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:21:55 2019

@author: deeperthought
"""


import pandas as pd
import os

PATH = '/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/Liz_Mary/Malignant_segmentor_nr2/'

folds = [PATH + x for x in os.listdir(PATH)]



TRAIN = [x for x in folds if 'train' in x]
VAL = [x for x in folds if 'val' in x]
TEST = [x for x in folds if 'test' in x]




LABELS = [x for x in folds if 'label' in x]
LABELS_TRAIN = [x for x in LABELS if 'train' in x][0]
LABELS_VAL = [x for x in LABELS if 'val' in x][0]
LABELS_TEST = [x for x in LABELS if 'test' in x][0]

index_to_remove_train = []
index_to_remove_test = []
index_to_remove_val = []

BLACKLIST = pd.read_csv(LABELS_TRAIN, header=None)
index_to_remove_train.extend(BLACKLIST.loc[BLACKLIST[0].str.contains('LABELS_SEGMENTATION/LABELS/')].index)

BLACKLIST = pd.read_csv(LABELS_TEST, header=None)
index_to_remove_test.extend(BLACKLIST.loc[BLACKLIST[0].str.contains('LABELS_SEGMENTATION/LABELS/')].index)

BLACKLIST = pd.read_csv(LABELS_VAL, header=None)
index_to_remove_val.extend(BLACKLIST.loc[BLACKLIST[0].str.contains('LABELS_SEGMENTATION/LABELS/')].index)


for l in TRAIN:
    print(l)
    scans = pd.read_csv(l, header=None)
    scans = scans.drop(index=index_to_remove_train) 
    name = l.replace('.txt','_onlyLiz.txt')
    scans.to_csv(name, index=False, header=False)



for l in VAL:
    print(l)
    scans = pd.read_csv(l, header=None)
    scans = scans.drop(index=index_to_remove_val) 
    name = l.replace('.txt','_onlyLiz.txt')
    scans.to_csv(name, index=False, header=False)    

for l in TEST:
    print(l)
    scans = pd.read_csv(l, header=None)
    scans = scans.drop(index=index_to_remove_test) 
    name = l.replace('.txt','_onlyLiz.txt')
    scans.to_csv(name, index=False, header=False)