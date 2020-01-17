# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:35:34 2019

@author: hirsch
"""

import os
import pandas as pd


HDD = '/media/hirsch/RNN_training/alignedNii/'

LABELS_PATH = '/home/hirsch/Documents/projects/MSKCC/FINAL_TRAIN_DATA/LABELS/'

ALL_TRIPLES_CSV = '/media/hirsch/RNN_training/All_Triples.csv'

WORKING_DIR = '/media/hirsch/RNN_training/'


pathology_csv = pd.read_csv(ALL_TRIPLES_CSV)

pathology_csv['Side'] 

pathology_csv['Exam_side'] = pathology_csv['ID_date'] + '_' + pathology_csv['Side']

exams = os.listdir(HDD)


scans_HDD = [os.path.join(d, x)
            for d, dirs, files in os.walk(HDD)
            for x in files if x.endswith(".nii")]


len(scans_HDD)

#malignant_scans_csv = pathology_csv.loc[pathology_csv['Pathology'] == 'Malignant', ['ID_date', 'Side']]
#benign_scans_csv = pathology_csv.loc[pathology_csv['Pathology'] == 'Benign', ['ID_date', 'Side']]
#len(malignant_scans_csv)


scans_HDD = pd.DataFrame(scans_HDD, columns = ['file'])

scans_HDD['Side'] = scans_HDD['file'].apply(lambda x : 'Right' if x.split('.nii')[0][-1] == 'r' else 'Left' )

scans_HDD['ID_date'] = scans_HDD['file'].apply(lambda x : x.split('/')[-2])

scans_HDD['Exam_side'] = scans_HDD['ID_date'] + '_' + scans_HDD['Side']



malignant_scans_HDD =  scans_HDD.loc[scans_HDD['Exam_side'].isin(pathology_csv.loc[pathology_csv['Pathology'] == 'Malignant', 'Exam_side'])]

len(set(malignant_scans_HDD['ID_date']))


benign_scans = list(set(scans_HDD['Exam_side']) - set(malignant_scans_HDD['Exam_side']))

benign_scans_HDD =  scans_HDD.loc[~ scans_HDD['Exam_side'].isin(malignant_scans_HDD['Exam_side'])]
len(set(benign_scans_HDD['ID_date']))

#benign_scans_HDD.to_csv(WORKING_DIR + 'benign_scans_HDD.csv', index=False)

Benigns_t2 = benign_scans_HDD['file'].apply(lambda x : x if x.split('/')[-1].startswith('t2') else 0)
Benigns_t2 = Benigns_t2[Benigns_t2 != 0]

Benigns_t1pre = benign_scans_HDD['file'].apply(lambda x : x if x.split('/')[-1].startswith('t1pre') else 0)
Benigns_t1pre = Benigns_t1pre[Benigns_t1pre != 0]

Benigns_t1post = benign_scans_HDD['file'].apply(lambda x : x if x.split('/')[-1].startswith('t1post') else 0)
Benigns_t1post = Benigns_t1post[Benigns_t1post != 0]


Benigns_t2 = Benigns_t2.sort_values(ascending=True)
Benigns_t1pre = Benigns_t1pre.sort_values(ascending=True)
Benigns_t1post = Benigns_t1post.sort_values(ascending=True)


assert len(Benigns_t1post) == len(Benigns_t1pre)
assert len(Benigns_t1post) == len(Benigns_t2)

# Check for order
assert [x.split('/')[-2] for x in Benigns_t2] == [x.split('/')[-2] for x in Benigns_t1pre]
assert [x.split('/')[-2] for x in Benigns_t2] == [x.split('/')[-2] for x in Benigns_t1post]



#Benigns_t2.to_csv(WORKING_DIR + 'benigns_t2.txt', index=False)
#Benigns_t1pre.to_csv(WORKING_DIR + 'benigns_t1pre.txt', index=False)
#Benigns_t1post.to_csv(WORKING_DIR + 'benigns_t1post.txt', index=False)


labeled_exams = [ 'MSKCC' + x.split('_label')[0].split('MSKCC')[-1] for x in os.listdir(LABELS_PATH)]

Malignants_with_target = malignant_scans_HDD.loc[malignant_scans_HDD['ID_date'].isin(labeled_exams)]

Malignant_unlabeled = malignant_scans_HDD.loc[~ malignant_scans_HDD['ID_date'].isin(labeled_exams)]

#Malignants_with_target.to_csv(WORKING_DIR + 'Malignants_with_target.csv', index=False)

Malignants_t2 = Malignants_with_target['file'].apply(lambda x : x if x.split('/')[-1].startswith('t2') else 0)
Malignants_t2 = Malignants_t2[Malignants_t2 != 0]

Malignants_t1pre = Malignants_with_target['file'].apply(lambda x : x if x.split('/')[-1].startswith('t1pre') else 0)
Malignants_t1pre = Malignants_t1pre[Malignants_t1pre != 0]

Malignants_t1post = Malignants_with_target['file'].apply(lambda x : x if x.split('/')[-1].startswith('t1post') else 0)
Malignants_t1post = Malignants_t1post[Malignants_t1post != 0]

labeled_exams = pd.DataFrame(labeled_exams, columns = ['ID_date'])
labeled_exams['file'] = os.listdir(LABELS_PATH)
labeled_exams_out = labeled_exams.loc[labeled_exams['ID_date'].isin(Malignants_with_target['ID_date']), 'file']
labeled_exams_out = LABELS_PATH + labeled_exams_out


Malignants_t2 = Malignants_t2.sort_values(ascending=True)
Malignants_t1pre = Malignants_t1pre.sort_values(ascending=True)
Malignants_t1post = Malignants_t1post.sort_values(ascending=True)
labeled_exams_out = labeled_exams_out.sort_values(ascending=True)


assert len(Malignants_t1post) == len(Malignants_t1pre)
assert len(Malignants_t1post) == len(Malignants_t2)
assert len(Malignants_t1post) == len(labeled_exams_out)

assert [x.split('/')[-2] for x in Malignants_t2] == [x.split('/')[-2] for x in Malignants_t1pre]
assert [x.split('/')[-2] for x in Malignants_t2] == [x.split('/')[-2] for x in Malignants_t1post]
assert [x.split('/')[-2] for x in Malignants_t2] == ['MSK' + x.split('/')[-1].split('_label')[0].split('MSK')[-1] for x in labeled_exams_out]



Malignants_t2.to_csv(WORKING_DIR + 'malignants_t2.txt', index=False)
Malignants_t1pre.to_csv(WORKING_DIR + 'malignants_t1pre.txt', index=False)
Malignants_t1post.to_csv(WORKING_DIR + 'malignants_t1post.txt', index=False)
labeled_exams_out.to_csv(WORKING_DIR + 'malignant_target_labels.txt', index=False)


Malignants_unlabeled = malignant_scans_HDD.loc[~ malignant_scans_HDD['ID_date'].isin(labeled_exams)]





