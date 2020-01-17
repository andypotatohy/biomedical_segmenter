#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 15:40:35 2019

@author: deeperthought
"""

import os
import nibabel as nib
import numpy as np
import multiprocessing        
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool   
import pandas as pd

subs = '/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/kirby_data/train_sub.txt'
t1c = '/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/kirby_data/train_t1post.txt'

T1post_scans_qualityCheck = '/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/kirby_data_info_t1post.csv'
Subs_qualityCheck = '/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/kirby_data_info_subs.csv'


blacklist = {}

def get_scan_info(scan):
    nii = nib.load(scan)
    img = nii.get_data()
    res = np.diag(nii.affine)
    shape = img.shape
    STD = np.std(img)
    return([scan, res[0], res[1], res[2], shape[0],shape[1],shape[2], STD])


if not os.path.exists(T1post_scans_qualityCheck):

    t1c_scans = [x[:-1] for x in open(t1c).readlines()]
    pool = Pool(multiprocessing.cpu_count() - 2 )
    t1_scans_info = pool.map(get_scan_info, t1c_scans, chunksize=250)
    pool.close()
    pool.join()    
    df = pd.DataFrame(t1_scans_info, columns = ['Scan','Res_x','Res_y','Res_z','Shape_x','Shape_y','Shape_z','Std'])
    df.to_csv(T1post_scans_qualityCheck, index=False)

else:
    df = pd.read_csv(T1post_scans_qualityCheck)
    
    # 2D images:
    df = df.sort_values(['Shape_x'])
    df[['Scan', 'Shape_x']].head(10)
    blacklist['planar_images'] = df.loc[df['Shape_x'] < 5, 'Scan'].values
    
    # No contrast (noise images) based on standard deviation:
    df = df.sort_values(['Std'])
    df[['Scan', 'Std']].head(10)    
    blacklist['no contrast'] = df.loc[df['Std'] < 1,'Scan'].values

    # Weird resolution
    df = df.sort_values(['Res_x'], ascending=True)
    df[['Scan', 'Res_x']].head(10)    
    blacklist['Suspicious coronal resolution'] = df.loc[df['Res_x'] < 1,'Scan'].values
    
    blacklist_scanID = []
    for key in blacklist.keys():
        blacklist_scanID.extend(blacklist[key])
        
    blacklist_scanID = pd.DataFrame(blacklist_scanID)
    blacklist_scanID[0].to_csv('/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/Blacklist.csv', index=False)

    
if not os.path.exists(Subs_qualityCheck):  
    subs_scans = [x[:-1] for x in open(subs).readlines()]
    pool = Pool(multiprocessing.cpu_count() - 2 )
    subs_scans_info = pool.map(get_scan_info, subs_scans, chunksize=250)
    pool.close()
    pool.join()    
    subs_scans_df = pd.DataFrame(subs_scans_info, columns = ['Scan','Res_x','Res_y','Res_z','Shape_x','Shape_y','Shape_z','Std'])
    subs_scans_df.to_csv(Subs_qualityCheck, index=False)
    

else:
    subs_scans_df = pd.read_csv(Subs_qualityCheck)
    
    # 2D images:
    subs_scans_df = subs_scans_df.sort_values(['Shape_x'])
    subs_scans_df[['Scan', 'Shape_x']].head(10)    

    # No contrast (noise images) based on standard deviation:
    subs_scans_df = subs_scans_df.sort_values(['Std'])
    subs_scans_df[['Scan', 'Std']].head(10)       
    # Weird resolution
    subs_scans_df = subs_scans_df.sort_values(['Res_x'], ascending=False)
    subs_scans_df[['Scan', 'Res_x']].head(10)    
    subs_scans_df = subs_scans_df.sort_values(['Res_y'], ascending=False)
    subs_scans_df[['Scan', 'Res_y']].head(10)   
    subs_scans_df = subs_scans_df.sort_values(['Res_z'], ascending=False)
    subs_scans_df[['Scan', 'Res_z']].head(10)       
