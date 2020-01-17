#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:45:32 2019

@author: deeperthought
"""

import os
import pandas as pd

breastmask_path = '/home/deeperthought/kirby_MSK/BreastMasks/alignedNii-Aug2019/'

masks = [breastmask_path + x for x in os.listdir(breastmask_path)]

breastMasks_df = pd.DataFrame()
breastMasks_df[0] = masks
breastMasks_df['exam'] = breastMasks_df[0].apply(lambda x : x.split('/')[-1].split('_T1')[0])
breastMasks_df['side'] = breastMasks_df[0].apply(lambda x : x.split('/')[-1].split('T1_')[1].split('_post')[0])
breastMasks_df['scan_ID'] = breastMasks_df['exam'] + '_' + breastMasks_df['side'] 




train_labels = '/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/CV_alignedNii-Aug2019_actual-F4-training/train_labels_1.txt'
train_sub = '/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/CV_alignedNii-Aug2019_actual-F4-training/train_sub_1.txt'
train_t1post = '/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/CV_alignedNii-Aug2019_actual-F4-training/train_t1post_1.txt'

val_t1post = '/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/CV_alignedNii-Aug2019_actual-F4-training/val_t1post.txt'
val_sub = '/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/CV_alignedNii-Aug2019_actual-F4-training/val_sub.txt'
val_labels = '/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/CV_alignedNii-Aug2019_actual-F4-training/val_labels.txt'

train_labels = pd.read_csv(train_labels, header=None)
train_sub = pd.read_csv(train_sub, header=None)
train_t1post = pd.read_csv(train_t1post, header=None)

val_t1post = pd.read_csv(val_t1post, header=None)
val_sub = pd.read_csv(val_sub, header=None)
val_labels = pd.read_csv(val_labels, header=None)


train_t1post[0][0]
train_t1post['exam'] = train_t1post[0].apply(lambda x : x.split('/')[-2])
train_t1post['side'] = train_t1post[0].apply(lambda x : x.split('/')[-1].split('T1_')[-1].split('_post')[0])
train_t1post['scan_ID'] = train_t1post['exam'] + '_' + train_t1post['side']

val_t1post['exam'] = val_t1post[0].apply(lambda x : x.split('/')[-2])
val_t1post['side'] = val_t1post[0].apply(lambda x : x.split('/')[-1].split('T1_')[-1].split('_post')[0])
val_t1post['scan_ID'] = val_t1post['exam'] + '_' + val_t1post['side']


scans_in_trainSet = train_t1post.loc[train_t1post['scan_ID'].isin(breastMasks_df['scan_ID'])].index
breastMasks_in_trainSet = breastMasks_df.loc[breastMasks_df['scan_ID'].isin(train_t1post['scan_ID'])].index
scans_not_in_trainSet = train_t1post.loc[~ train_t1post['scan_ID'].isin(breastMasks_df['scan_ID'])].index

scans_in_valSet = val_t1post.loc[val_t1post['scan_ID'].isin(breastMasks_df['scan_ID'])].index
breastMasks_in_valSet = breastMasks_df.loc[breastMasks_df['scan_ID'].isin(val_t1post['scan_ID'])].index
scans_not_in_valSet = val_t1post.loc[~ val_t1post['scan_ID'].isin(breastMasks_df['scan_ID'])].index



OUTPUT_PATH = '/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/CV_alignedNii-Aug2019-BreastMask/'

train_t1post.iloc[scans_in_trainSet][0].to_csv(OUTPUT_PATH + 'train_t1post.txt', header=False, index=False)
train_sub.iloc[scans_in_trainSet][0].to_csv(OUTPUT_PATH + 'train_sub.txt', header=False, index=False)
train_labels.iloc[scans_in_trainSet][0].to_csv(OUTPUT_PATH + 'train_labels.txt', header=False, index=False)
#breastMasks_df[0].iloc[breastMasks_in_trainSet].sort_values().to_csv(OUTPUT_PATH + 'train_breastMask.txt', header=False, index=False)

val_t1post.iloc[scans_in_valSet][0].to_csv(OUTPUT_PATH + 'val_t1post.txt', header=False, index=False)
val_sub.iloc[scans_in_valSet][0].to_csv(OUTPUT_PATH + 'val_sub.txt', header=False, index=False)
val_labels.iloc[scans_in_valSet][0].to_csv(OUTPUT_PATH + 'val_labels.txt', header=False, index=False)
#breastMasks_df[0].iloc[breastMasks_in_valSet].sort_values().to_csv(OUTPUT_PATH + 'val_breastMask.txt', header=False, index=False)
