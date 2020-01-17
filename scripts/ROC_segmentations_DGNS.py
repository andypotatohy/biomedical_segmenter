# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:21:07 2019

@author: hirsch
"""

from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, LeaveOneOut, StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn import svm
from sklearn.neural_network import MLPClassifier

save_fig = False
filter_BIRADS = True
USE_EDGES_TABLE = False
use_diagnostic_only = False 
use_screening_only = True
compare_without_demographics = False
BIRADS_Benign = [1,2,3]
BIRADS_Malignant = [4,5,6]

use_risk_adjusted_screening = False
TIME_INTERVAL = 0.5
CUMULATIVE_TIME_INTERVAL = False# so far only counts 0.5 + 1 year in advance.
use_cleaned_spreadsheet = True
generate_artificial_data = True # generates dummy plots for comparison

WD = '/home/hirsch/Documents/projects/Breast_segmentation/DeepPriors_package/training_sessions/'
RESULT_NAME = 'Classification_from_segmentation_ROC_Model174_likelihoods'
# MultiPriors_MSKCC_fullHeadSegmentation_configFile_DM_2019-03-11_1846  /  MultiPriors_MSKCC_fullHeadSegmentation_configFile_DM_subs_2019-04-25_1322
RESULTS_TABLE_PATH = WD + 'MultiPriors_MSKCC_fullHeadSegmentation_configFile_DM_2019-03-11_1846/connected_components_regression/'
# SEGMENTATIONS_SUMMARY_epoch174_likelihoods_3  , SEGMENTATIONS_SUMMARY_epoch65_likelihoods
RESULTS_TABLE_NAME = 'SEGMENTATIONS_SUMMARY_epoch174_likelihoods_3.csv'

if filter_BIRADS:
  RESULT_NAME += '_BIRADS_{}-{}'.format(BIRADS_Benign, BIRADS_Malignant)
if use_diagnostic_only:
  RESULT_NAME += '_Diagnostic'  
if use_screening_only:
  RESULT_NAME += '_Screening'
if use_risk_adjusted_screening:
  RESULT_NAME += '_RiskAdj-{}'.format(TIME_INTERVAL)
if CUMULATIVE_TIME_INTERVAL:
  RESULT_NAME += '_cumulative'
if use_cleaned_spreadsheet:
  RESULT_NAME += '_clean'
    

ALL_TRIPLES_FUTURE_LABELS =  '/media/hirsch/RNN_training/All_Triples_Cleaned_FutureLabels.csv'
  


df = pd.read_csv(RESULTS_TABLE_PATH + RESULTS_TABLE_NAME)

# Remove faulty scans:
sessions_identical_images = pd.read_csv('/home/hirsch/Documents/projects/MSKCC/Data_spreadsheets/sessionWithIdenticalImages.csv')
sessions_identical_images.columns = ['session']
df['session'] = df['scan_ID'].apply(lambda x : x[:-2])

df = df.loc[~ df['session'].isin(sessions_identical_images['session'])]

# Add other features to table. From 'Diamond' Spreadsheet. Get subject age, match with scan_ID.
clinical = pd.read_excel('/home/hirsch/Documents/projects/MSKCC/Data_spreadsheets/Diamond_and_Gold/CCNY_CLINICAL_4_17_2019.xlsx', header=[0,1])

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
  
  feat['ETHNICITY'] = ethn.values
  feat['RACE'] = race.values
  feat['ETHNICITY'].value_counts()
  feat['DE-ID'] = race.index
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

def add_family_hx(df, clinical):
  fam = pd.DataFrame(columns=['DE-ID','Family Hx'])
  fam['Family Hx'] = clinical['Family Hx']['Family Hx']
  fam['Family Hx'] = fam['Family Hx'].apply(lambda x : 1 if x == 'Yes' else 0)
  fam['DE-ID'] = clinical['Family Hx']['Family Hx'].index
  fam.reset_index(level=0, inplace=True) 
  df2 = df.copy()
  df2['ID_date'] = df2['scan_ID'].apply(lambda x : x[:-2])
  df2['DE-ID'] = df2['ID_date'].apply(lambda x : x[:-9]) 
  df3 = df2.merge(fam, on=['DE-ID'])
  df3.head()
  df3 = df3[['scan_ID','Family Hx']]
  df4 = df3.merge(df, on=['scan_ID'])
  df4 = df4.loc[df4['scan_ID'].isin(df['scan_ID'])]
  return df4

if use_diagnostic_only:
  all_triples = pd.read_csv(ALL_TRIPLES_FUTURE_LABELS)
  all_triples = all_triples[['scan_ID','Population']]
  df = df.merge(all_triples, on=['scan_ID'])
  df = df.loc[df['Population'] == 'Diagnostic']

if use_screening_only:  
  all_triples = pd.read_csv(ALL_TRIPLES_FUTURE_LABELS)
  all_triples = all_triples[['scan_ID','Population']]
  df = df.merge(all_triples, on=['scan_ID'])
  df = df.loc[df['Population'] == 'Screening']

df = add_age(df, clinical)
df = add_ethnicity(df, clinical)
df = add_family_hx(df, clinical)


print(df['pathology'].value_counts())

np.unique(df['BIRADS'])
BIRADS_Benign = [str(x) for x in BIRADS_Benign]
BIRADS_Malignant = [str(x) for x in BIRADS_Malignant]

if filter_BIRADS:
  BIRADS_keep = BIRADS_Benign + BIRADS_Malignant 
  df = df.loc[df['BIRADS'].isin(BIRADS_keep) ]
  print('Selecting BIRADS. DataFrame only contains {}'.format(np.unique(df['BIRADS'])))

if use_risk_adjusted_screening:
  risk_adj = pd.read_csv(ALL_TRIPLES_FUTURE_LABELS)
  df_yr = risk_adj[['scan_ID', 'Pathology', '{} years away'.format(TIME_INTERVAL)]]

  if CUMULATIVE_TIME_INTERVAL:
    df_yr1 = risk_adj[['scan_ID', 'Pathology', '0.5 years away']]
    df_yr2 = risk_adj[['scan_ID', 'Pathology', '1 years away']]
    df_yr = df_yr1.merge(df_yr2, on=['scan_ID', 'Pathology'])
    def foo(row):
      if (row['0.5 years away'] == 'Malignant' or row['1 years away'] == 'Malignant'):
        return 'Malignant'
      elif (row['0.5 years away'] == 'Benign' or row['1 years away'] == 'Benign'):
        return 'Benign'
    
    df_yr['1 years away'] = df_yr.apply(foo, axis=1)
    
        
  mal_scans = df_yr.loc[df_yr['{} years away'.format(TIME_INTERVAL)] == 'Malignant','scan_ID']
  df = df.loc[df['scan_ID'].isin(df_yr['scan_ID'])]
  df = df.merge(df_yr, on = ['scan_ID']).drop_duplicates()
  
  df = df.loc[df['{} years away'.format(TIME_INTERVAL)] != 'Unknown']
  df = df.loc[~ ((df['pathology'] == 'Malignant') * (df['{} years away'.format(TIME_INTERVAL)] == 'Malignant') )]
  # remove also benigns that were previously malignants (post OP ? )
  df = df.loc[~ ((df['pathology'] == 'Malignant') * (df['{} years away'.format(TIME_INTERVAL)] == 'Benign') )]

  
  
  df['pathology'] = df['{} years away'.format(TIME_INTERVAL)]
  # Drop rows with 'unknown'
  #df['pathology'].value_counts()
  df = df.loc[df['pathology'].isin(['Benign', 'Malignant'])]
  print(df['pathology'].value_counts())
  print('############# Malignants ################')
  print(df.loc[df['pathology'] == 'Malignant','BIRADS'].value_counts())
  print('############# Benigns ################')  
  print(df.loc[df['pathology'] == 'Benign','BIRADS'].value_counts())
  
  #print(df.loc[df['pathology'] == 'Malignant',['scan_ID','BIRADS']])
  
  
  
  
df['GT'] = df['pathology'].apply(lambda x : 1 if x == 'Malignant' else 0)
df['GT'].value_counts()

df['n_total_pixels_log'] = np.log(df['n_total_pixels'] +1)
df['size_largest_component_log'] = np.log(df['size_largest_component']+1)

X = df[['n_total_pixels_log', 'n_connected_components','size_largest_component_log','Age', 'ETHNICITY','RACE', 'Family Hx']]
y = df[['GT']]
X = X.values
X.shape
y = y.values
# RobustScaler
classifier = make_pipeline(StandardScaler(),
                           LogisticRegression(C=1,random_state=1, class_weight = {0:1, 1:20}))            
cv = StratifiedKFold(n_splits=6)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
y_pred_full = [0]
y_true_full = [0]

for train, test in cv.split(X, y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    y_pred = [list(x) for x in probas_]
    y_store = [list(x)[0] for x in  y[test]]
    #y_scans.extend()
    # add scans    
    y_pred_full.extend(y_pred)
    y_true_full.extend(y_store)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3)
             #label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
    
plt.figure(figsize=(4,4))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
print('ROC-AUC : {}'.format(mean_auc))
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'MRI + demo AUC = %0.2f $\pm$ %0.2f' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#if filter_BIRADS:
#  plt.title('Negatives vs Positives \n BIRADS {} {}'.format(' '.join(BIRADS_Benign), ' '.join(BIRADS_Malignant)))
#else:
#  plt.title('DGNS segmentation')
plt.legend(loc="lower right")
plt.show()
#if save_fig:
#  plt.savefig(RESULTS_TABLE_PATH + RESULT_NAME)
#####################################

if compare_without_demographics:
  
  X = df[['n_total_pixels_log', 'n_connected_components','size_largest_component_log']]
  y = df[['GT']]
  X = X.values
  
  cv = StratifiedKFold(n_splits=6)
  tprs = []
  aucs = []
  mean_fpr = np.linspace(0, 1, 100)
  i = 0
#  y_pred_full = [0]
#  y_true_full = [0]
  y = y.values
  classifier2 = make_pipeline(RobustScaler(),
                             LogisticRegression(C=1,random_state=1))
  for train, test in cv.split(X, y):
      probas_ = classifier2.fit(X[train], y[train]).predict_proba(X[test])
      y_pred = [list(x) for x in probas_]
      y_store = [list(x)[0] for x in  y[test]]
#      y_pred_full.extend(y_pred)
#      y_true_full.extend(y_store)
      # Compute ROC curve and area the curve
      fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
      tprs.append(interp(mean_fpr, fpr, tpr))
      tprs[-1][0] = 0.0
      roc_auc = auc(fpr, tpr)
      aucs.append(roc_auc)
      i += 1
  mean_tpr = np.mean(tprs, axis=0)
  mean_tpr[-1] = 1.0
  mean_auc = auc(mean_fpr, mean_tpr)
  print('ROC-AUC : {}'.format(mean_auc))
  std_auc = np.std(aucs)
  plt.plot(mean_fpr, mean_tpr, color='red',
           label=r'MRI AUC = %0.3f $\pm$ %0.2f' % (mean_auc, std_auc),
           lw=2, alpha=.5)
  std_tpr = np.std(tprs, axis=0)
  tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
  tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
  #plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
  #                 label=r'$\pm$ 1 std. dev.')
  plt.legend(loc="lower right")
  plt.show()

if save_fig:
  plt.savefig(RESULTS_TABLE_PATH + RESULT_NAME + '.png')


#%%         RISK ADJUSTED SCREENING & EARLY DETECTION : PPV AND NPV



####################### Histograms ###############################
radiologist_PPV  = 0.26    # At a prevalence of 
radiologist_NPV =  0.98
myBins = np.linspace(0,1,num=2000)

y_pred_full = y_pred_full[1:]
y_true_full = y_true_full[1:]
M = [y_pred_full[x[0]][1] for x in np.argwhere(np.array(y_true_full) > 0)]
B = [y_pred_full[x[0]][1] for x in np.argwhere(np.array(y_true_full) == 0)]

B = np.random.choice(B, size=int(len(M)/0.028), replace=True)


M.sort()
B.sort()
Mh = np.histogram(M, bins=myBins)
Bh = np.histogram(B, bins=myBins)
######################### NPV ###################################
myX = []
myY = []
Benign = 0
Malignant = 0
Omitted_biopsy = 0
TN = []
for i in range(len(Mh[0])):
  Benign += Bh[0][i]
  Malignant += Mh[0][i]
  TN.append(Benign)
  Omitted_biopsy = (Benign + Malignant)
  myX.append(Mh[1][i])
  myY.append(Benign/float(Omitted_biopsy))

TN.append(np.sum(Bh[0]))
myX.append(1)
myY.append(float(Benign)/(Benign+Malignant))  

X_indx = [myY.index(x) for x in myY if np.abs(x - radiologist_NPV) < 0.01][-1]
NPV_Meeting_point = myX[X_indx]
NPV_Number_cases_meeting_point  = TN[X_indx]
######################### PPV ###################################
myX2 = []
myY2 = []
Benign = 0
Malignant = 0
Biopsied = 0
TP = []
for i in np.arange(len(Mh[0])-1,0,-1):
  Benign += Bh[0][i]
  Malignant += Mh[0][i]
  TP.append(Malignant)
  Biopsied = (Benign + Malignant) # This is wrong. Not all benigns were sent to biopsy !!! Only BIRADS 4,5
  myX2.append(Mh[1][i])
  myY2.append(Malignant/float(Biopsied))

TP.append(np.sum(Mh[0]))
myX2.append(0)
myY2.append(float(Malignant)/(Benign+Malignant))  

X_indx = [myY2.index(x) for x in myY2 if np.abs(x - radiologist_PPV) < 1][0]
PPV_Meeting_point = myX2[X_indx]
PPV_Number_cases_meeting_point  = TP[X_indx]

######################## plot ######################################  
FIGSIZE = (12,3.5)  
FONTSIZE = 14
xlimit = max(M[-1],B[-1])
fig, [ax1, ax2, ax3] = plt.subplots(nrows=1, ncols=3,figsize=FIGSIZE)

weights = np.ones_like(M)/float(len(M))
ax1.hist(M,100, alpha=0.9, color='red')
ax1.set_title('6 month prediction')
ax1.set_ylabel('# MRI exams', fontsize=FONTSIZE)
ax1.set_xlabel('Predicted Risk', fontsize=FONTSIZE)
weights = np.ones_like(B)/float(len(B))
ax1.hist(B, 100, alpha=0.5, color='green')
ax1.set_xlim([-0.005,xlimit])

ax1.set_yscale('log')
 
 # NPV
ax22 = ax2.twinx()
ax2.plot(myX,myY, 'k-')
ax2.axhline(radiologist_NPV, color='g', linestyle='dashed')
#plt.vlines(x=NPV_Meeting_point,ymin=0,ymax=NPV_Number_cases_meeting_point, color='r', linestyle='dashed')
ax22.plot(myX,TN, 'b-')
ax2.set_title('MRI at 12 months')
ax2.set_ylabel('Negative Predictive Value', color='k', fontsize=FONTSIZE)
ax22.set_ylabel('# Avoided Exams', color='b', fontsize=FONTSIZE)
ax2.set_xlabel('Predicted Risk', fontsize=FONTSIZE)
ax2.set_xlim([-0.005,xlimit])
ax2.spines['right'].set_color('blue')
ax22.yaxis.label.set_color('blue')
ax22.tick_params(axis='y', colors='blue')
ax2.grid(linewidth=0.5, alpha = 0.5)
ax22.grid(linewidth=0.5, alpha = 0.5)
ax2.yaxis.grid(False)  
  
 # PPV 

ax3.plot(myX2,myY2, 'k-')

ax3.axhline(radiologist_PPV, color='g', linestyle='dashed')
ax33 = ax3.twinx()
#plt.vlines(x=PPV_Meeting_point,ymin=0,ymax=PPV_Number_cases_meeting_point, colors='r', linestyles='dashed')
ax33.plot(myX2,TP, 'b-')
ax3.grid(linewidth=0.5, alpha = 0.5)
ax33.grid(linewidth=0.5, alpha = 0.5)
ax3.yaxis.grid(False)
ax3.set_title('Biopsy today')
ax3.set_xlabel('Predicted Risk', fontsize=FONTSIZE)
ax3.set_ylabel('Positive Predictive Value', color='k', fontsize=FONTSIZE)
ax33.set_ylabel('# Early Detections', color='b', fontsize=FONTSIZE)
ax3.set_xlim([-0.005,xlimit])

ax3.spines['right'].set_color('blue')
ax33.yaxis.label.set_color('blue')
ax33.tick_params(axis='y', colors='blue')
plt.tight_layout()
if save_fig:
  plt.savefig(RESULTS_TABLE_PATH + '/PPV_NPV_Real_{}.png'.format(RESULT_NAME))  
   

#%%        Artificial data

if generate_artificial_data: 
#  M_dummy = np.random.normal(loc=0.6, scale=0.1, size=len(M)*2)
#  B_dummy = np.random.normal(loc=0.4, scale=0.09, size=len(B)) 
  M_dummy = np.random.triangular(left=0.2, mode=0.6, right=1, size=106*100)
  B_dummy = np.random.triangular(left=0, mode=0.4, right=0.8,  size=3500*100) 
  M_dummy.sort()
  B_dummy.sort()
#  M_dummy = M_dummy[15:-15]  
#  B_dummy = B_dummy[15:-15] 
  Mh = np.histogram(M_dummy, bins=myBins)
  Bh = np.histogram(B_dummy, bins=myBins)
  Mh = tuple([np.array([x/100. for x in  Mh[0]]), Mh[1]])
  Bh = tuple([np.array([x/100. for x in  Bh[0]]), Bh[1]])
  ######################### NPV ###################################
  myX = []
  myY = []
  Benign = 0
  Malignant = 0
  Omitted_biopsy = 0
  TN = []
  for i in range(len(Mh[0])):
    Benign += Bh[0][i]
    Malignant += Mh[0][i]
    TN.append(Benign)
    Omitted_biopsy = (Benign + Malignant)
    myX.append(Mh[1][i])
    myY.append(Benign/float(Omitted_biopsy))
  
  TN.append(np.sum(Bh[0]))
  myX.append(1)
  myY.append(float(Benign)/(Benign+Malignant))  
  ######################### PPV ###################################
  myX2 = []
  myY2 = []
  Benign = 0
  Malignant = 0
  Biopsied = 0
  TP = []
  for i in np.arange(len(Mh[0])-1,0,-1):
    Benign += Bh[0][i]
    Malignant += Mh[0][i]
    TP.append(Malignant)
    Biopsied = (Benign + Malignant) # This is wrong. Not all benigns were sent to biopsy !!! Only BIRADS 4,5
    myX2.append(Mh[1][i])
    myY2.append(Malignant/float(Biopsied))
 
  TP.append(np.sum(Mh[0]))
  myX2.append(0)
  myY2.append(float(Malignant)/(Benign+Malignant))  
    
  ######################## plot ######################################  
  FONTSIZE = 14
  xlimit = 1
  fig, [ax1, ax2, ax3] = plt.subplots(nrows=1, ncols=3,figsize=FIGSIZE)
  
  weights = np.ones_like(M_dummy)/float(len(M_dummy))
  ax1.hist(M_dummy,100, alpha=0.5, color='red')
  weights = np.ones_like(B_dummy)/float(len(B_dummy))
  ax1.hist(B_dummy, 100, alpha=0.5, color='green')
  ax1.set_xlim([-0.005,xlimit])
  ax1.set_title('1 year prediction')
  ax1.set_ylabel('# MRI exams', fontsize=FONTSIZE)
  ax1.set_xlabel('Predicted Risk', fontsize=FONTSIZE)
  ax1.set_yscale('log')
  # = [item.get_text() for item in ax1.get_yticklabels()]

  ax1.tick_params(axis=u'y', which=u'both',length=0)
  ax1.set_yticklabels([0,0,1,10,100])
  plt.show()
  
  ax22 = ax2.twinx()
  ax2.plot(myX,myY, 'k-')
  #ax2.title('MRI at 12 months')
  #ax2.axhline(radiologist_NPV, color='y', linestyle='dashed')
  ax22.plot(myX,TN, 'b-')
  ax2.set_title('MRI at 2 years')
  ax2.set_ylabel('Negative Predictive Value', color='k', fontsize=FONTSIZE)
  ax22.set_ylabel('# Avoided Exams', color='b', fontsize=FONTSIZE)
  ax2.set_xlabel('Predicted Risk', fontsize=FONTSIZE)
  
  ax2.set_xlim([-0.005,xlimit])
  
  ax2.spines['right'].set_color('blue')
  ax22.yaxis.label.set_color('blue')
  ax22.tick_params(axis='y', colors='blue')
  ax2.grid(linewidth=0.5, alpha = 0.5)
  ax22.grid(linewidth=0.5, alpha = 0.5)
  ax2.yaxis.grid(False)  
    
  ax33 = ax3.twinx()
  ax3.set_title('Biopsy today')
  ax3.set_xlabel('Predicted Risk', fontsize=FONTSIZE)
  ax3.set_ylabel('Positive Predictive Value', color='k', fontsize=FONTSIZE)
  ax33.set_ylabel('# Early Detections', color='b', fontsize=FONTSIZE)
  ax3.plot(myX2,myY2, 'k-')
  #ax3.axhline(radiologist_PPV, color='y', linestyle='dashed')
  ax33.plot(myX2,TP, 'b-')
  ax3.grid(linewidth=0.5, alpha = 0.5)
  ax33.grid(linewidth=0.5, alpha = 0.5)
  ax3.yaxis.grid(False)
  ax3.set_xlim([-0.005,xlimit])
  ax3.spines['right'].set_color('blue')
  ax33.yaxis.label.set_color('blue')
  ax33.tick_params(axis='y', colors='blue')
  #ax33.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
  plt.tight_layout()  
  if save_fig:
    plt.savefig(RESULTS_TABLE_PATH + '/PPV_NPV_Dummy_{}.png'.format(RESULT_NAME))  
    
    
    
