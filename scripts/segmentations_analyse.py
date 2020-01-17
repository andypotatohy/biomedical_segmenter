# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:11:46 2019

@author: hirsch
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

save_plot = True
filter_BIRADS = True
USE_EDGES_TABLE = False
use_diagnostic_only = False
use_screening_only = False
BIRADS_Benign = [1,2,3]
BIRADS_Malignant = [4,5,6]

WD = '/home/hirsch/Documents/projects/Breast_segmentation/DeepPriors_package/training_sessions/'
RESULTS_TABLE_PATH = WD + 'MultiPriors_MSKCC_fullHeadSegmentation_configFile_DM_subs_2019-04-25_1322/connected_components_regression/'
RESULTS_TABLE_NAME = 'SEGMENTATIONS_SUMMARY_epoch65_likelihoods.csv'
#RESULTS_TABLE_NAME_EDGES = 'SEGMENTATIONS_SUMMARY_epoch174_edges.csv'
RESULT_IMAGE_NAME = 'Segmentation_analysis_model65_Age_SCREENING.png'

df = pd.read_csv(RESULTS_TABLE_PATH + RESULTS_TABLE_NAME)
if USE_EDGES_TABLE :
  df_edges = pd.read_csv(RESULTS_TABLE_PATH + RESULTS_TABLE_NAME_EDGES)
  df_edges = df_edges[['scan_ID', 'n_total_pixels', 'n_connected_components', 'size_largest_component']]
  df_edges.columns = ['scan_ID', 'n_total_pixels_edges', 'n_connected_components_edges', 'size_largest_component_edges']
  df = pd.merge(df, df_edges, on=['scan_ID'])

def add_age(df, clinical):
  ages = clinical['Unnamed: 0_level_0']  
  ages['DE-ID'] = ages.index
  ages.reset_index(level=0, inplace=True)
  ages = ages[['DE-ID','DOB']]
  df2 = df.copy()
  df2['ID_date'] = df2['scan_ID'].apply(lambda x : x[:-2])
  df2['DE-ID'] = df2['ID_date'].apply(lambda x : x[:-9])
  df3 = df2.merge(ages, on=['DE-ID'])
  df3.head()
  df3['Age'] = df3.apply(lambda row : int(row['ID_date'][-8:-4]) - int(row['DOB']), axis=1)
  df3 = df3[['scan_ID','Age']]
  df4 = df3.merge(df, on=['scan_ID'])
  df4 = df4.loc[df4['scan_ID'].isin(df['scan_ID'])]
  return df4

clinical = pd.read_excel('/home/hirsch/Documents/projects/MSKCC/Data_spreadsheets/Diamond_and_Gold/CCNY_CLINICAL_4_17_2019.xlsx', header=[0,1])
  
df = add_age(df, clinical)

def add_ethnicity(df, clinical):
  clinical.columns
  ethn = clinical['Unnamed: 3_level_0']['ETHNICITY']
  race = clinical['Unnamed: 2_level_0']['RACE']
  feat = pd.DataFrame(columns=['ETHNICITY', 'RACE'])
  race[race == 'WHITE'] = 3
  race[race == 'BLACK OR AFRICAN AMERICAN'] = 2
  race[race == 'ASIAN-FAR EAST/INDIAN SUBCONT'] = 1
  race[~ race.isin([1,2,3])] = 0
  
  ethn[ethn == 'HISPANIC OR LATINO'] = 1
  ethn[ethn == 'NOT HISPANIC'] = -1
  ethn[~ ethn.isin([-1,1])] = 0
  
  feat['ETHNICITY'] = ethn
  feat['RACE'] = race
  feat['ETHNICITY'].value_counts()
  feat['DE-ID'] = feat.index
  feat.reset_index(level=0, inplace=True)
  feat = feat[['DE-ID','ETHNICITY','RACE']]  
  df2 = df.copy()
  df2['ID_date'] = df2['scan_ID'].apply(lambda x : x[:-2])
  df2['DE-ID'] = df2['ID_date'].apply(lambda x : x[:-9]) 
  df3 = df2.merge(feat, on=['DE-ID'])
  df3.head()

  df3 = df3[['scan_ID','ETHNICITY','RACE']]
  df4 = df3.merge(df, on=['scan_ID'])
  df4 = df4.loc[df4['scan_ID'].isin(df['scan_ID'])]
  return df4

df = add_ethnicity(df, clinical)
df.columns

## evaluate the histogram
#values, base = np.histogram(df['Age'], bins=100)
##evaluate the cumulative
#cumulative = np.cumsum(values)
## plot the cumulative function
#plt.plot(base[:-1], cumulative, c='blue')
#plt.title('Cumulative Age')
#plt.ylabel('# Patients with age <= x')
#plt.xlabel('Age')
#plt.savefig('/home/hirsch/Documents/projects/Breast_segmentation/DeepPriors_package/training_sessions/MultiPriors_MSKCC_fullHeadSegmentation_configFile_DM_2019-03-11_1846/connected_components_regression/Age_cumsum.png')


if use_diagnostic_only:
  all_triples = pd.read_csv('/media/hirsch/RNN_training/All_Triples.csv')
  all_triples = all_triples[['scan_ID','Population']]
  df2 = df.merge(all_triples, on=['scan_ID'])
  df = df2[['scan_ID', 'Age', 'n_total_pixels', 'n_connected_components', 'size_largest_component', 'BIRADS', 'pathology', 'Population']]
  df = df.loc[df['Population'] == 'Diagnostic']
  
if use_screening_only:  
  all_triples = pd.read_csv('/media/hirsch/RNN_training/All_Triples.csv')
  all_triples = all_triples[['scan_ID','Population']]
  df2 = df.merge(all_triples, on=['scan_ID'])
  df = df2[['scan_ID', 'Age', 'n_total_pixels', 'n_connected_components', 'size_largest_component', 'BIRADS', 'pathology', 'Population']]
  df = df.loc[df['Population'] == 'Screening']  
  
np.unique(df['BIRADS'])

if filter_BIRADS:
  BIRADS_keep = BIRADS_Benign + BIRADS_Malignant
  BIRADS_keep = [str(x) for x in BIRADS_keep]
  df = df.loc[df['BIRADS'].isin(BIRADS_keep) ]
  print('Selecting BIRADS. DataFrame only contains {}'.format(np.unique(df['BIRADS'])))

df['n_total_pixels_log'] = np.log(df['n_total_pixels'] + 1)
df['n_connected_components_log'] = np.log(df['n_connected_components'] + 1)
df['size_largest_component_log'] = np.log(df['size_largest_component'] + 1)

if USE_EDGES_TABLE :
  df['n_total_pixels_edges_log'] = np.log(df['n_total_pixels_edges'] + 1)
  df['n_connected_components_edges_log'] = np.log(df['n_connected_components_edges'] + 1)
  df['size_largest_component_edges_log'] = np.log(df['size_largest_component_edges'] + 1)

df.describe()
pathology = df.pathology.unique()
if filter_BIRADS:
  RESULT_IMAGE_NAME = RESULT_IMAGE_NAME.split('.png')[0] + '_' + ','.join(str(x) for x in BIRADS_Benign) + '_vs_' + ','.join(str(x) for x in BIRADS_Malignant) + '.png'


plt.figure(figsize=(15,7))
plt.subplot(2,3,1)
#plt.title('n_total_pixels_log')
sns.distplot(df.loc[df['pathology'] == 'Benign','n_total_pixels_log'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkgreen', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
             
sns.distplot(df.loc[df['pathology'] == 'Malignant','n_total_pixels_log'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkred', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})      
plt.legend(('Benign','Malignant'))
             
plt.subplot(2,3,2)
#plt.title('n_connected_components')
sns.distplot(df.loc[df['pathology'] == 'Benign','n_connected_components'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkgreen', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
             
sns.distplot(df.loc[df['pathology'] == 'Malignant','n_connected_components'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkred', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})      
plt.legend(('Benign','Malignant'))
             
plt.subplot(2,3,3)
#plt.title('size_largest_component_log')
sns.distplot(df.loc[df['pathology'] == 'Benign','size_largest_component_log'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkgreen', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
             
sns.distplot(df.loc[df['pathology'] == 'Malignant','size_largest_component_log'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkred', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})    
plt.legend(('Benign','Malignant'))


plt.subplot(2,3,4)
#plt.title('n_total_pixels_edges_log')
sns.distplot(df.loc[df['pathology'] == 'Benign','Age'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkgreen', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
             
sns.distplot(df.loc[df['pathology'] == 'Malignant','Age'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkred', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})    
plt.legend(('Benign','Malignant'))







if save_plot:
  plt.savefig(RESULTS_TABLE_PATH +RESULT_IMAGE_NAME )



