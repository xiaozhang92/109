---
title:  standardize
notebook: models.ipynb
nav_include: 2
---

## Contents
{:.no_toc}
*  
{: toc}





```python
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

import numpy as np
from __future__ import division
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV
import sklearn as sk
import csv
import random
```


### Inputs



```python
the_dir = r'./ADNI'

files_to_join = ['ADNIMERGE.csv','UPENNBIOMK_MASTER.csv']#'UCD_ADNI2_WMH_10_26_15.csv']

fields_to_delete = ['PTID','SITE','EXAMDATE']

out_fn = ''
```




```python
def read_file(fn):
    print ('Reading {}...'.format(fn))
    df = pd.read_csv(fn)
    # if VISCODE2 exists, use VISCODE2
    if 'VISCODE2' in df.columns:
        del df['VISCODE']
        df = df.rename(columns={"VISCODE2": "VISCODE"})
        #print (df[df[['RID','VISCODE']].duplicated()]) # to check duplicates
    return df
```




```python
for fn in files_to_join:
    fn_dir = '{}/{}'.format(the_dir, fn)
    df = read_file(fn_dir)
    
    if fn == files_to_join[0]:
        df_all = df.copy()
    else:
        df_all = pd.merge(df_all, df, how='inner', on=['RID', 'VISCODE'])
    print ('{} columns included.'.format(len(df_all.columns)))
    print ('{} samples remaining.'.format(len(df_all.index)))

df_all.head()
```


    Reading ./ADNI/ADNIMERGE.csv...
    94 columns included.
    13017 samples remaining.
    Reading ./ADNI/UPENNBIOMK_MASTER.csv...
    106 columns included.
    5869 samples remaining.





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RID</th>
      <th>PTID</th>
      <th>VISCODE</th>
      <th>SITE</th>
      <th>COLPROT</th>
      <th>ORIGPROT</th>
      <th>EXAMDATE</th>
      <th>DX_bl</th>
      <th>AGE</th>
      <th>PTGENDER</th>
      <th>...</th>
      <th>STDS</th>
      <th>DRWDTE</th>
      <th>RUNDATE</th>
      <th>ABETA</th>
      <th>TAU</th>
      <th>PTAU</th>
      <th>ABETA_RAW</th>
      <th>TAU_RAW</th>
      <th>PTAU_RAW</th>
      <th>update_stamp_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>011_S_0003</td>
      <td>bl</td>
      <td>11</td>
      <td>ADNI1</td>
      <td>ADNI1</td>
      <td>9/12/05</td>
      <td>AD</td>
      <td>81.3</td>
      <td>Male</td>
      <td>...</td>
      <td>167082</td>
      <td>2005-09-12</td>
      <td>2007-10-26</td>
      <td>131.0</td>
      <td>68.0</td>
      <td>21.0</td>
      <td>131.0</td>
      <td>68.0</td>
      <td>21.0</td>
      <td>2016-07-06 16:15:51.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>011_S_0003</td>
      <td>bl</td>
      <td>11</td>
      <td>ADNI1</td>
      <td>ADNI1</td>
      <td>9/12/05</td>
      <td>AD</td>
      <td>81.3</td>
      <td>Male</td>
      <td>...</td>
      <td>181286</td>
      <td>2005-09-12</td>
      <td>2008-11-12</td>
      <td>132.0</td>
      <td>54.9</td>
      <td>19.8</td>
      <td>149.0</td>
      <td>55.5</td>
      <td>12.8</td>
      <td>2016-07-06 16:15:51.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>011_S_0003</td>
      <td>bl</td>
      <td>11</td>
      <td>ADNI1</td>
      <td>ADNI1</td>
      <td>9/12/05</td>
      <td>AD</td>
      <td>81.3</td>
      <td>Male</td>
      <td>...</td>
      <td>ALL</td>
      <td>2005-09-12</td>
      <td>2016-03-09</td>
      <td>131.0</td>
      <td>61.4</td>
      <td>20.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-07-06 16:15:51.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>011_S_0003</td>
      <td>m12</td>
      <td>11</td>
      <td>ADNI1</td>
      <td>ADNI1</td>
      <td>9/12/06</td>
      <td>AD</td>
      <td>81.3</td>
      <td>Male</td>
      <td>...</td>
      <td>181286</td>
      <td>2006-09-13</td>
      <td>2008-11-12</td>
      <td>137.0</td>
      <td>76.5</td>
      <td>21.1</td>
      <td>155.0</td>
      <td>77.5</td>
      <td>13.7</td>
      <td>2016-07-06 16:15:51.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>011_S_0003</td>
      <td>m12</td>
      <td>11</td>
      <td>ADNI1</td>
      <td>ADNI1</td>
      <td>9/12/06</td>
      <td>AD</td>
      <td>81.3</td>
      <td>Male</td>
      <td>...</td>
      <td>ALL</td>
      <td>2006-09-13</td>
      <td>2016-03-09</td>
      <td>137.0</td>
      <td>76.5</td>
      <td>21.1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-07-06 16:15:51.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 106 columns</p>
</div>





```python
print (df_all.columns[:20])
print (df_all.columns[21:])
```


    Index(['RID', 'PTID', 'VISCODE', 'SITE', 'COLPROT', 'ORIGPROT', 'EXAMDATE',
           'DX_bl', 'AGE', 'PTGENDER', 'PTEDUCAT', 'PTETHCAT', 'PTRACCAT',
           'PTMARRY', 'APOE4', 'FDG', 'PIB', 'AV45', 'CDRSB', 'ADAS11'],
          dtype='object')
    Index(['MMSE', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting',
           'RAVLT_perc_forgetting', 'FAQ', 'MOCA', 'EcogPtMem', 'EcogPtLang',
           'EcogPtVisspat', 'EcogPtPlan', 'EcogPtOrgan', 'EcogPtDivatt',
           'EcogPtTotal', 'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat', 'EcogSPPlan',
           'EcogSPOrgan', 'EcogSPDivatt', 'EcogSPTotal', 'FLDSTRENG', 'FSVERSION',
           'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform',
           'MidTemp', 'ICV', 'DX', 'EXAMDATE_bl', 'CDRSB_bl', 'ADAS11_bl',
           'ADAS13_bl', 'MMSE_bl', 'RAVLT_immediate_bl', 'RAVLT_learning_bl',
           'RAVLT_forgetting_bl', 'RAVLT_perc_forgetting_bl', 'FAQ_bl',
           'FLDSTRENG_bl', 'FSVERSION_bl', 'Ventricles_bl', 'Hippocampus_bl',
           'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl', 'ICV_bl',
           'MOCA_bl', 'EcogPtMem_bl', 'EcogPtLang_bl', 'EcogPtVisspat_bl',
           'EcogPtPlan_bl', 'EcogPtOrgan_bl', 'EcogPtDivatt_bl', 'EcogPtTotal_bl',
           'EcogSPMem_bl', 'EcogSPLang_bl', 'EcogSPVisspat_bl', 'EcogSPPlan_bl',
           'EcogSPOrgan_bl', 'EcogSPDivatt_bl', 'EcogSPTotal_bl', 'FDG_bl',
           'PIB_bl', 'AV45_bl', 'Years_bl', 'Month_bl', 'Month', 'M',
           'update_stamp_x', 'BATCH', 'KIT', 'STDS', 'DRWDTE', 'RUNDATE', 'ABETA',
           'TAU', 'PTAU', 'ABETA_RAW', 'TAU_RAW', 'PTAU_RAW', 'update_stamp_y'],
          dtype='object')


### Feature Engineering



```python

fields_to_delete = ['DX_bl','PTID','SITE','COLPROT','ORIGPROT','EXAMDATE','EXAMDATE_bl',
                    'BATCH','DRWDTE','RUNDATE','update_stamp_y','KIT','STDS','update_stamp_x',
                    'FLDSTRENG_bl','FSVERSION_bl','CDRSB','CDRSB_bl','MMSE','MMSE_bl','FAQ',
                    'FAQ_bl','RAVLT_immediate','RAVLT_immediate_bl','RAVLT_learning',
                    'RAVLT_learning_bl','RAVLT_perc_forgetting_bl','RAVLT_perc_forgetting',
                    'RAVLT_forgetting', 'RAVLT_forgetting_bl']

for field in fields_to_delete:
    del df_all[field]

print ('{} columns included.'.format(len(df_all.columns)))
```


    76 columns included.




```python
remeasure_cols = ['Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV']
for col in remeasure_cols:
    df_all[str(col+'_pct_delta')] = (df_all[col] - df_all[str(col+'_bl')])/df_all[str(col+'_bl')]
```




```python
df_ob_cols = df_all.select_dtypes(include=[object])
for col in df_ob_cols.columns:
    print (col,df_ob_cols[col].unique())
```


    VISCODE ['bl' 'm12' 'm24' 'm48' 'm60' 'm36' 'm72' 'm84']
    PTGENDER ['Male' 'Female']
    PTETHCAT ['Not Hisp/Latino' 'Hisp/Latino' 'Unknown']
    PTRACCAT ['White' 'Black' 'Asian' 'More than one' 'Am Indian/Alaskan' 'Unknown'
     'Hawaiian/Other PI']
    PTMARRY ['Married' 'Widowed' 'Divorced' 'Never married' 'Unknown']
    FLDSTRENG ['1.5 Tesla MRI' '3 Tesla MRI' nan]
    FSVERSION ['Cross-Sectional FreeSurfer (FreeSurfer Version 4.3)'
     'Cross-Sectional FreeSurfer (5.1)' nan]
    DX ['Dementia' 'MCI' 'CN']




```python
fields_one_hot = ['PTGENDER','PTETHCAT','PTRACCAT','PTMARRY','FLDSTRENG','FSVERSION']
df_all_with_dummy = pd.get_dummies(df_all, columns=fields_one_hot, dummy_na=True, drop_first=True)
df_all_with_dummy.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RID</th>
      <th>VISCODE</th>
      <th>AGE</th>
      <th>PTEDUCAT</th>
      <th>APOE4</th>
      <th>FDG</th>
      <th>PIB</th>
      <th>AV45</th>
      <th>ADAS11</th>
      <th>ADAS13</th>
      <th>...</th>
      <th>PTRACCAT_nan</th>
      <th>PTMARRY_Married</th>
      <th>PTMARRY_Never married</th>
      <th>PTMARRY_Unknown</th>
      <th>PTMARRY_Widowed</th>
      <th>PTMARRY_nan</th>
      <th>FLDSTRENG_3 Tesla MRI</th>
      <th>FLDSTRENG_nan</th>
      <th>FSVERSION_Cross-Sectional FreeSurfer (FreeSurfer Version 4.3)</th>
      <th>FSVERSION_nan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>bl</td>
      <td>81.3</td>
      <td>18</td>
      <td>1.0</td>
      <td>1.09079</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22.0</td>
      <td>31.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>bl</td>
      <td>81.3</td>
      <td>18</td>
      <td>1.0</td>
      <td>1.09079</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22.0</td>
      <td>31.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>bl</td>
      <td>81.3</td>
      <td>18</td>
      <td>1.0</td>
      <td>1.09079</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22.0</td>
      <td>31.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>m12</td>
      <td>81.3</td>
      <td>18</td>
      <td>1.0</td>
      <td>1.10384</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24.0</td>
      <td>35.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>m12</td>
      <td>81.3</td>
      <td>18</td>
      <td>1.0</td>
      <td>1.10384</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24.0</td>
      <td>35.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 98 columns</p>
</div>





```python
df_all.isnull().sum()
```





    RID                         0
    VISCODE                     0
    AGE                         0
    PTGENDER                    0
    PTEDUCAT                    0
    PTETHCAT                    0
    PTRACCAT                    0
    PTMARRY                     0
    APOE4                       0
    FDG                      1817
    PIB                      5615
    AV45                     3224
    ADAS11                     16
    ADAS13                     59
    MOCA                     3176
    EcogPtMem                3165
    EcogPtLang               3169
    EcogPtVisspat            3185
    EcogPtPlan               3169
    EcogPtOrgan              3236
    EcogPtDivatt             3185
    EcogPtTotal              3167
    EcogSPMem                3167
    EcogSPLang               3167
    EcogSPVisspat            3211
    EcogSPPlan               3195
    EcogSPOrgan              3279
    EcogSPDivatt             3234
    EcogSPTotal              3171
    FLDSTRENG                 745
                             ... 
    EcogPtOrgan_bl           3397
    EcogPtDivatt_bl          3356
    EcogPtTotal_bl           3338
    EcogSPMem_bl             3346
    EcogSPLang_bl            3346
    EcogSPVisspat_bl         3384
    EcogSPPlan_bl            3370
    EcogSPOrgan_bl           3456
    EcogSPDivatt_bl          3403
    EcogSPTotal_bl           3350
    FDG_bl                   1702
    PIB_bl                   5812
    AV45_bl                  3353
    Years_bl                    0
    Month_bl                    0
    Month                       0
    M                           0
    ABETA                      19
    TAU                        86
    PTAU                       29
    ABETA_RAW                2189
    TAU_RAW                  2241
    PTAU_RAW                 2181
    Ventricles_pct_delta      469
    Hippocampus_pct_delta    1224
    WholeBrain_pct_delta      333
    Entorhinal_pct_delta     1349
    Fusiform_pct_delta       1349
    MidTemp_pct_delta        1349
    ICV_pct_delta             202
    Length: 83, dtype: int64





```python
fields_to_delete = ['PTRACCAT_Unknown']
for col in fields_to_delete:
    del df_all_with_dummy[col]
```




```python
%matplotlib inline
print (len(df_all_with_dummy.index))
a = df_all_with_dummy.isnull().sum()
#print (len(a[a<1200]))
df_all_with_dummy = df_all_with_dummy[a[a<1200].index]
print (len(df_all_with_dummy.columns))
a = df_all_with_dummy.isnull().sum()
a.hist(bins=20)
```


    5869
    54





    <matplotlib.axes._subplots.AxesSubplot at 0x11084cdd8>




![png](models_files/models_14_2.png)




```python
df_all_with_dummy.columns
```





    Index(['RID', 'VISCODE', 'AGE', 'PTEDUCAT', 'APOE4', 'ADAS11', 'ADAS13',
           'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform',
           'MidTemp', 'ICV', 'DX', 'ADAS11_bl', 'ADAS13_bl', 'Ventricles_bl',
           'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl',
           'MidTemp_bl', 'ICV_bl', 'Years_bl', 'Month_bl', 'Month', 'M', 'ABETA',
           'TAU', 'PTAU', 'Ventricles_pct_delta', 'WholeBrain_pct_delta',
           'ICV_pct_delta', 'PTGENDER_Male', 'PTGENDER_nan',
           'PTETHCAT_Not Hisp/Latino', 'PTETHCAT_Unknown', 'PTETHCAT_nan',
           'PTRACCAT_Asian', 'PTRACCAT_Black', 'PTRACCAT_Hawaiian/Other PI',
           'PTRACCAT_More than one', 'PTRACCAT_White', 'PTRACCAT_nan',
           'PTMARRY_Married', 'PTMARRY_Never married', 'PTMARRY_Unknown',
           'PTMARRY_Widowed', 'PTMARRY_nan', 'FLDSTRENG_3 Tesla MRI',
           'FLDSTRENG_nan',
           'FSVERSION_Cross-Sectional FreeSurfer (FreeSurfer Version 4.3)',
           'FSVERSION_nan'],
          dtype='object')





```python
df_all_with_dummy['y'] = 0
df_all_with_dummy.loc[df_all_with_dummy['DX']=='Dementia','y']=1
df_all_with_dummy.y.hist()

```





    <matplotlib.axes._subplots.AxesSubplot at 0x1107de8d0>




![png](models_files/models_16_1.png)




```python
import numpy as np
```




```python
print (len(df_all_with_dummy['RID'].unique()))
df_grp = df_all_with_dummy.groupby('RID')


df_train = []
df_test = []
for idx, grp in df_grp:
    if np.random.choice([0,1], p=[0.2, 0.8]) == 1:
        try:
            #print (grp)
            df_train = df_train.append(grp)
        except:
            df_train = grp.copy()
    else:
        try: 
            df_test = df_test.append(grp)
        except:
            df_test = grp.copy()

print (len(df_train.index))
print (len(df_test.index))
#df_all_with_dummy_standardized = df_all_with_dummy - df_all_with_dummy.mean()

#df_all_with_dummy
```


    1249
    4577
    1282




```python
y_train = df_train.y
y_test = df_test.y

del df_train['y']
del df_train['DX']
del df_test['y']
del df_test['DX']
```




```python
df_train.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RID</th>
      <th>VISCODE</th>
      <th>AGE</th>
      <th>PTEDUCAT</th>
      <th>APOE4</th>
      <th>ADAS11</th>
      <th>ADAS13</th>
      <th>Ventricles</th>
      <th>Hippocampus</th>
      <th>WholeBrain</th>
      <th>...</th>
      <th>PTRACCAT_nan</th>
      <th>PTMARRY_Married</th>
      <th>PTMARRY_Never married</th>
      <th>PTMARRY_Unknown</th>
      <th>PTMARRY_Widowed</th>
      <th>PTMARRY_nan</th>
      <th>FLDSTRENG_3 Tesla MRI</th>
      <th>FLDSTRENG_nan</th>
      <th>FSVERSION_Cross-Sectional FreeSurfer (FreeSurfer Version 4.3)</th>
      <th>FSVERSION_nan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>bl</td>
      <td>67.5</td>
      <td>10</td>
      <td>0.0</td>
      <td>14.33</td>
      <td>21.33</td>
      <td>39605.0</td>
      <td>6869.0</td>
      <td>1154980.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4</td>
      <td>bl</td>
      <td>67.5</td>
      <td>10</td>
      <td>0.0</td>
      <td>14.33</td>
      <td>21.33</td>
      <td>39605.0</td>
      <td>6869.0</td>
      <td>1154980.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4</td>
      <td>bl</td>
      <td>67.5</td>
      <td>10</td>
      <td>0.0</td>
      <td>14.33</td>
      <td>21.33</td>
      <td>39605.0</td>
      <td>6869.0</td>
      <td>1154980.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4</td>
      <td>m12</td>
      <td>67.5</td>
      <td>10</td>
      <td>0.0</td>
      <td>15.00</td>
      <td>22.00</td>
      <td>38527.0</td>
      <td>6451.0</td>
      <td>1117390.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4</td>
      <td>m12</td>
      <td>67.5</td>
      <td>10</td>
      <td>0.0</td>
      <td>15.00</td>
      <td>22.00</td>
      <td>38527.0</td>
      <td>6451.0</td>
      <td>1117390.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 53 columns</p>
</div>





```python
X_train = df_train.iloc[:,2:]
X_test = df_test.iloc[:,2:]
X_train.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>PTEDUCAT</th>
      <th>APOE4</th>
      <th>ADAS11</th>
      <th>ADAS13</th>
      <th>Ventricles</th>
      <th>Hippocampus</th>
      <th>WholeBrain</th>
      <th>Entorhinal</th>
      <th>Fusiform</th>
      <th>...</th>
      <th>PTRACCAT_nan</th>
      <th>PTMARRY_Married</th>
      <th>PTMARRY_Never married</th>
      <th>PTMARRY_Unknown</th>
      <th>PTMARRY_Widowed</th>
      <th>PTMARRY_nan</th>
      <th>FLDSTRENG_3 Tesla MRI</th>
      <th>FLDSTRENG_nan</th>
      <th>FSVERSION_Cross-Sectional FreeSurfer (FreeSurfer Version 4.3)</th>
      <th>FSVERSION_nan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>67.5</td>
      <td>10</td>
      <td>0.0</td>
      <td>14.33</td>
      <td>21.33</td>
      <td>39605.0</td>
      <td>6869.0</td>
      <td>1154980.0</td>
      <td>3983.0</td>
      <td>19036.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>67.5</td>
      <td>10</td>
      <td>0.0</td>
      <td>14.33</td>
      <td>21.33</td>
      <td>39605.0</td>
      <td>6869.0</td>
      <td>1154980.0</td>
      <td>3983.0</td>
      <td>19036.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>67.5</td>
      <td>10</td>
      <td>0.0</td>
      <td>14.33</td>
      <td>21.33</td>
      <td>39605.0</td>
      <td>6869.0</td>
      <td>1154980.0</td>
      <td>3983.0</td>
      <td>19036.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>67.5</td>
      <td>10</td>
      <td>0.0</td>
      <td>15.00</td>
      <td>22.00</td>
      <td>38527.0</td>
      <td>6451.0</td>
      <td>1117390.0</td>
      <td>3519.0</td>
      <td>18691.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>67.5</td>
      <td>10</td>
      <td>0.0</td>
      <td>15.00</td>
      <td>22.00</td>
      <td>38527.0</td>
      <td>6451.0</td>
      <td>1117390.0</td>
      <td>3519.0</td>
      <td>18691.0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 51 columns</p>
</div>





```python
X_train_stand = (X_train - X_train.mean())/(X_train.max() - X_train.min())
X_test_stand = (X_test - X_train.mean())/(X_train.max() - X_train.min())
```




```python
X_train_stand.fillna(0,inplace=True)
X_test_stand.fillna(0,inplace=True)
```




```python
print (X_train_stand.shape)
print (X_test_stand.shape)
print (y_train.shape)
print (y_test.shape)

X_train_stand.to_csv('./data/X_train_null_1200.csv',index=False)
X_test_stand.to_csv('./data/X_test_null_1200.csv',index=False)
y_train.to_csv('./data/y_train_null_1200.csv',index=False)
y_test.to_csv('./data/y_test_null_1200.csv',index=False)
```


    (4577, 51)
    (1282, 51)
    (4577,)
    (1282,)


### Model Building

#### RF



```python
from itertools import product
from collections import OrderedDict
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
```




```python
### baseline RF model
rf = RandomForestClassifier(random_state = 100)
rf.fit(X_train_stand, y_train)
```





    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=10, n_jobs=1, oob_score=False, random_state=100,
                verbose=0, warm_start=False)





```python
print(rf.score(X_test_stand, y_test))
```


    0.86895475819




```python
### ROC for baseline model
fpr1, tpr1, thres1 = roc_curve(y_test, rf.predict_proba(X_test_stand)[:,1])
fpr2, tpr2, thres2 = roc_curve(y_test, np.zeros((len(y_test), 1)))
auc1 = roc_auc_score(y_test, rf.predict_proba(X_test_stand)[:,1])
auc2 = roc_auc_score(y_test, np.zeros((len(y_test), 1)))

fig, ax = plt.subplots(1,1)
ax.plot(fpr1, tpr1, '-', alpha=0.8, label='Baseline RF model (AUC=%.2f)' % (auc1))
ax.plot(fpr2, tpr2, '-', alpha=0.8, label='All 0 classifier (AUC=%.2f)' % (auc2))
plt.title("ROC curves")
plt.legend()
```





    <matplotlib.legend.Legend at 0x1eae0c187b8>




![png](models_files/models_30_1.png)




```python
print('Baseline RF model gives AUC of {:.3f}'.format(roc_auc_score(y_test, rf.predict_proba(X_test_stand)[:,1])))
```


    Baseline RF model gives AUC of 0.898




```python
### param tuning
depth = [1,2,5,10,20,50,100,None]
ns = [10, 20, 50, 100, 150, 200]

accs = pd.DataFrame(np.zeros((len(depth)*len(ns), 3)))
accs.columns = ['max_depth','n_est','AUC']

i = 0
kf = KFold(n_splits=5)
for d in depth:
    for n in ns:
        print("depth:{}, n:{}".format(d,n))
        acc_cv = []
        
        for train, val in kf.split(X_train_stand):
            train_X, train_y, val_X, val_y = X_train_stand.iloc[train,:], pd.DataFrame(y_train).iloc[train,:], X_train_stand.iloc[val,:], pd.DataFrame(y_train).iloc[val,:]
            rf_temp = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=100)
            rf_temp.fit(train_X, train_y.values.ravel())
            acc_cv.append(roc_auc_score(val_y, rf_temp.predict_proba(val_X)[:,1]))

        accs.iloc[i,:] = [d, n, np.mean(acc_cv)]
        i+=1
```


    depth:1, n:10
    depth:1, n:20
    depth:1, n:50
    depth:1, n:100
    depth:1, n:150
    depth:1, n:200
    depth:2, n:10
    depth:2, n:20
    depth:2, n:50
    depth:2, n:100
    depth:2, n:150
    depth:2, n:200
    depth:5, n:10
    depth:5, n:20
    depth:5, n:50
    depth:5, n:100
    depth:5, n:150
    depth:5, n:200
    depth:10, n:10
    depth:10, n:20
    depth:10, n:50
    depth:10, n:100
    depth:10, n:150
    depth:10, n:200
    depth:20, n:10
    depth:20, n:20
    depth:20, n:50
    depth:20, n:100
    depth:20, n:150
    depth:20, n:200
    depth:50, n:10
    depth:50, n:20
    depth:50, n:50
    depth:50, n:100
    depth:50, n:150
    depth:50, n:200
    depth:100, n:10
    depth:100, n:20
    depth:100, n:50
    depth:100, n:100
    depth:100, n:150
    depth:100, n:200
    depth:None, n:10
    depth:None, n:20
    depth:None, n:50
    depth:None, n:100
    depth:None, n:150
    depth:None, n:200




```python
accs.sort_values(by = 'AUC', ascending=False).head()
```





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>max_depth</th>
      <th>n_est</th>
      <th>AUC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>5.0</td>
      <td>200.0</td>
      <td>0.931572</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5.0</td>
      <td>100.0</td>
      <td>0.930886</td>
    </tr>
    <tr>
      <th>16</th>
      <td>5.0</td>
      <td>150.0</td>
      <td>0.930589</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5.0</td>
      <td>50.0</td>
      <td>0.929270</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5.0</td>
      <td>20.0</td>
      <td>0.928254</td>
    </tr>
  </tbody>
</table>
</div>





```python
### best model
rf_best1 = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=100)
rf_best1.fit(X_train_stand, y_train)
rf_best1.score(X_test_stand, y_test)
```





    0.89079563182527299





```python
### ROC for best model
fpr1, tpr1, thres1 = roc_curve(y_test, rf_best1.predict_proba(X_test_stand)[:,1])
fpr2, tpr2, thres2 = roc_curve(y_test, np.zeros((len(y_test), 1)))
auc1 = roc_auc_score(y_test, rf_best1.predict_proba(X_test_stand)[:,1])
auc2 = roc_auc_score(y_test, np.zeros((len(y_test), 1)))
```




```python
print('Best RF model gives AUC of {:.3f}'.format(roc_auc_score(y_test, rf_best1.predict_proba(X_test_stand)[:,1])))
```


    Best RF model gives AUC of 0.907


### Xgboosting



```python
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np

X_train = pd.read_csv('X_train_null_1200.csv').as_matrix()
X_test = pd.read_csv('X_test_null_1200.csv').as_matrix()
y_train = pd.read_csv('y_train_null_1200.csv', header=None).as_matrix()
y_test = pd.read_csv('y_test_null_1200.csv', header=None).as_matrix()
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)

dtrain = xgb.DMatrix(X_train,label=y_train)
dtest = xgb.DMatrix(X_test,label=y_test)
```


    (5241, 50)
    (5241, 1)
    (628, 50)
    (628, 1)




```python





#xgboost_params
param = {}
param['booster']='gbtree'
param['objective'] = 'binary:logistic'
param['bst:eta'] = 0.04
param['seed']=  0
param['bst:max_depth'] = 7
param['silent'] =  1  
param['nthread'] = 12 # put more if you have
param['bst:subsample'] = 0.7
param['gamma'] = 1.0
param['colsample_bytree']= 1.0
param['num_parallel_tree']= 10
param['colsample_bylevel']= 0.7                  
param['lambda']=1
param['eval_metric'] = 'error'

evallist = [(dtest,'eval'),(dtrain,'train')]
watchlist = [(dtest,'eval')]



```




```python
num_round = 500
xgb.cv(param, dtrain, num_round, nfold=5,
       metrics={'error'}, seed=0, early_stopping_rounds = 50,
       callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
#bst.save_model('0001.model')
#bst.dump_model('dump0001.raw.txt')
```


    [0]	train-error:0.0543788+0.00564696	test-error:0.0774642+0.00977699
    [1]	train-error:0.0476534+0.00196409	test-error:0.0747924+0.0112659
    [2]	train-error:0.0431216+0.00269346	test-error:0.0675422+0.0110527
    [3]	train-error:0.0394486+0.00123362	test-error:0.0662064+0.0104792
    [4]	train-error:0.0354894+0.00101709	test-error:0.0658244+0.0106206
    [5]	train-error:0.0321504+0.0019513	test-error:0.060672+0.0104223
    [6]	train-error:0.0288114+0.00159212	test-error:0.0580018+0.00917999
    [7]	train-error:0.0265216+0.00210934	test-error:0.0549488+0.00990925
    [8]	train-error:0.024089+0.00109996	test-error:0.0530412+0.0100403
    [9]	train-error:0.0214652+0.000867452	test-error:0.0503696+0.00968534
    [10]	train-error:0.0193666+0.00199277	test-error:0.0471258+0.00918616
    [11]	train-error:0.0168382+0.00152774	test-error:0.0448364+0.00883216
    [12]	train-error:0.0155028+0.0019724	test-error:0.042356+0.00880593
    [13]	train-error:0.014358+0.00181832	test-error:0.0414028+0.00714506
    [14]	train-error:0.0132132+0.00154285	test-error:0.0400672+0.00643124
    [15]	train-error:0.0119254+0.00153111	test-error:0.0373962+0.00761212
    [16]	train-error:0.0102082+0.0015776	test-error:0.035679+0.00766381
    [17]	train-error:0.0090632+0.00135736	test-error:0.0333892+0.00859241
    [18]	train-error:0.0082522+0.00137081	test-error:0.0324354+0.00793149
    [19]	train-error:0.0073938+0.00136592	test-error:0.0312906+0.00812299
    [20]	train-error:0.0068688+0.000858711	test-error:0.030337+0.0078968
    [21]	train-error:0.005581+0.0008619	test-error:0.0293828+0.00747092
    [22]	train-error:0.004627+0.0008619	test-error:0.0267118+0.00746107
    [23]	train-error:0.0035774+0.000691515	test-error:0.024804+0.00807162
    [24]	train-error:0.0033868+0.000745444	test-error:0.0240404+0.00761622
    [25]	train-error:0.0028142+0.000697887	test-error:0.0225138+0.0076205
    [26]	train-error:0.0024326+0.000552392	test-error:0.0215604+0.00771707
    [27]	train-error:0.0020032+0.000467404	test-error:0.0198432+0.00653822
    [28]	train-error:0.001574+0.000387369	test-error:0.0185078+0.00617238
    [29]	train-error:0.0013832+0.000350441	test-error:0.0183168+0.00616519
    [30]	train-error:0.0010014+0.000486525	test-error:0.0171718+0.00624081
    [31]	train-error:0.000906+0.00038175	test-error:0.0164086+0.00564001
    [32]	train-error:0.0008582+0.000356981	test-error:0.0158366+0.00548438
    [33]	train-error:0.0006674+0.000316346	test-error:0.0148828+0.00555089
    [34]	train-error:0.0004768+0.000150841	test-error:0.0141196+0.00495432
    [35]	train-error:0.0003816+0.000116841	test-error:0.0139288+0.00461952
    [36]	train-error:0.0003816+0.000116841	test-error:0.0133568+0.0046751
    [37]	train-error:0.0003816+0.000116841	test-error:0.013929+0.00535047
    [38]	train-error:0.0003338+0.000116923	test-error:0.0137384+0.00538471
    [39]	train-error:0.0003338+0.000116923	test-error:0.01412+0.00534414
    [40]	train-error:0.000286+9.55008e-05	test-error:0.0135474+0.00472947
    [41]	train-error:0.000286+9.55008e-05	test-error:0.0137382+0.00477553
    [42]	train-error:0.0002384+0.000150841	test-error:0.0127842+0.00492524
    [43]	train-error:0.0002384+0.000150841	test-error:0.0127842+0.00492524
    [44]	train-error:0.0002384+0.000150841	test-error:0.0127842+0.00492524
    [45]	train-error:0.0002384+0.000150841	test-error:0.0131658+0.00495491
    [46]	train-error:0.0002384+0.000150841	test-error:0.012784+0.00403036
    [47]	train-error:0.0001906+9.53008e-05	test-error:0.0133564+0.00439376
    [48]	train-error:0.0001906+9.53008e-05	test-error:0.0129748+0.00437694
    [49]	train-error:0.0001906+9.53008e-05	test-error:0.012784+0.00384547
    [50]	train-error:0.0001906+9.53008e-05	test-error:0.0129748+0.00322784
    [51]	train-error:0.0001906+9.53008e-05	test-error:0.0133564+0.00325062
    [52]	train-error:0.0001906+9.53008e-05	test-error:0.0129748+0.00322784
    [53]	train-error:0.000143+0.00011676	test-error:0.0129748+0.00437694
    [54]	train-error:0.000143+0.00011676	test-error:0.0129748+0.00437694
    [55]	train-error:0.000143+0.00011676	test-error:0.0129748+0.00437694
    [56]	train-error:0.000143+0.00011676	test-error:0.0129748+0.00437694
    [57]	train-error:0.000143+0.00011676	test-error:0.0129748+0.00437694
    [58]	train-error:0.000143+0.00011676	test-error:0.0125932+0.00457206
    [59]	train-error:0.000143+0.00011676	test-error:0.0129748+0.00437694
    [60]	train-error:0.000143+0.00011676	test-error:0.0129748+0.00437694
    [61]	train-error:0.000143+0.00011676	test-error:0.0129748+0.00437694
    [62]	train-error:0.000143+0.00011676	test-error:0.0125932+0.00457206
    [63]	train-error:0.000143+0.00011676	test-error:0.0129748+0.00437694
    [64]	train-error:0.000143+0.00011676	test-error:0.0129748+0.00437694
    [65]	train-error:0.000143+0.00011676	test-error:0.0129748+0.00437694
    [66]	train-error:0.000143+0.00011676	test-error:0.0129748+0.00437694
    [67]	train-error:0.000143+0.00011676	test-error:0.0129748+0.00437694
    [68]	train-error:0.000143+0.00011676	test-error:0.0129748+0.00437694
    [69]	train-error:0.000143+0.00011676	test-error:0.0129748+0.00437694
    [70]	train-error:0.000143+0.00011676	test-error:0.0129748+0.00437694
    [71]	train-error:0.000143+0.00011676	test-error:0.0129748+0.00437694
    [72]	train-error:0.000143+0.00011676	test-error:0.0129748+0.00437694
    [73]	train-error:0.000143+0.00011676	test-error:0.0129748+0.00437694
    [74]	train-error:0.000143+0.00011676	test-error:0.0127838+0.00461992
    [75]	train-error:0.000143+0.00011676	test-error:0.0127838+0.00461992
    [76]	train-error:0.000143+0.00011676	test-error:0.0127838+0.00461992
    [77]	train-error:9.52e-05+0.000116596	test-error:0.012593+0.00465171
    [78]	train-error:9.52e-05+0.000116596	test-error:0.012593+0.00465171
    [79]	train-error:9.52e-05+0.000116596	test-error:0.012593+0.00465171
    [80]	train-error:9.52e-05+0.000116596	test-error:0.012593+0.00465171
    [81]	train-error:9.52e-05+0.000116596	test-error:0.012593+0.00465171
    [82]	train-error:9.52e-05+0.000116596	test-error:0.012593+0.00465171
    [83]	train-error:9.52e-05+0.000116596	test-error:0.012593+0.00465171
    [84]	train-error:9.52e-05+0.000116596	test-error:0.0129746+0.00469866
    [85]	train-error:9.52e-05+0.000116596	test-error:0.0129746+0.00469866
    [86]	train-error:9.52e-05+0.000116596	test-error:0.012593+0.00465171
    [87]	train-error:9.52e-05+0.000116596	test-error:0.012593+0.00465171
    [88]	train-error:9.52e-05+0.000116596	test-error:0.0129746+0.00469866
    [89]	train-error:9.52e-05+0.000116596	test-error:0.0129746+0.00469866
    [90]	train-error:9.52e-05+0.000116596	test-error:0.0129746+0.00469866
    [91]	train-error:9.52e-05+0.000116596	test-error:0.0129746+0.00469866
    [92]	train-error:9.52e-05+0.000116596	test-error:0.0129746+0.00469866
    [93]	train-error:9.52e-05+0.000116596	test-error:0.0129746+0.00469866
    [94]	train-error:9.52e-05+0.000116596	test-error:0.0129746+0.00469866
    [95]	train-error:9.52e-05+0.000116596	test-error:0.0129746+0.00469866
    [96]	train-error:9.52e-05+0.000116596	test-error:0.0129746+0.00469866
    [97]	train-error:9.52e-05+0.000116596	test-error:0.0129746+0.00469866
    [98]	train-error:9.52e-05+0.000116596	test-error:0.0129746+0.00469866
    [99]	train-error:9.52e-05+0.000116596	test-error:0.0129746+0.00469866
    [100]	train-error:9.52e-05+0.000116596	test-error:0.0129746+0.00469866
    [101]	train-error:9.52e-05+0.000116596	test-error:0.0129746+0.00469866
    [102]	train-error:9.52e-05+0.000116596	test-error:0.0129746+0.00469866
    [103]	train-error:9.52e-05+0.000116596	test-error:0.0129746+0.00469866
    [104]	train-error:9.52e-05+0.000116596	test-error:0.0129746+0.00469866
    [105]	train-error:9.52e-05+0.000116596	test-error:0.0129746+0.00469866
    [106]	train-error:9.52e-05+0.000116596	test-error:0.0129746+0.00469866
    [107]	train-error:9.52e-05+0.000116596	test-error:0.0124022+0.00431033
    [108]	train-error:9.52e-05+0.000116596	test-error:0.0127838+0.00437762
    [109]	train-error:9.52e-05+0.000116596	test-error:0.0127838+0.00437762
    [110]	train-error:9.52e-05+0.000116596	test-error:0.0127838+0.00437762
    [111]	train-error:9.52e-05+0.000116596	test-error:0.0127838+0.00437762
    [112]	train-error:9.52e-05+0.000116596	test-error:0.0127838+0.00437762
    [113]	train-error:9.52e-05+0.000116596	test-error:0.0127838+0.00437762
    [114]	train-error:4.76e-05+9.52e-05	test-error:0.012593+0.00432732
    [115]	train-error:4.76e-05+9.52e-05	test-error:0.0122112+0.0043271
    [116]	train-error:4.76e-05+9.52e-05	test-error:0.0122112+0.0043271
    [117]	train-error:0+0	test-error:0.0122112+0.0043271
    [118]	train-error:0+0	test-error:0.0122112+0.0043271
    [119]	train-error:0+0	test-error:0.0122112+0.0043271
    [120]	train-error:0+0	test-error:0.0122112+0.0043271
    [121]	train-error:0+0	test-error:0.0122112+0.0043271
    [122]	train-error:0+0	test-error:0.0122112+0.0043271
    [123]	train-error:0+0	test-error:0.0122112+0.0043271
    [124]	train-error:0+0	test-error:0.0122112+0.0043271
    [125]	train-error:0+0	test-error:0.0122112+0.0043271
    [126]	train-error:0+0	test-error:0.012593+0.00432732
    [127]	train-error:0+0	test-error:0.012593+0.00432732
    [128]	train-error:0+0	test-error:0.012593+0.00432732
    [129]	train-error:0+0	test-error:0.012593+0.00432732
    [130]	train-error:0+0	test-error:0.012593+0.00432732
    [131]	train-error:0+0	test-error:0.012593+0.00432732
    [132]	train-error:0+0	test-error:0.012593+0.00432732
    [133]	train-error:0+0	test-error:0.012593+0.00432732
    [134]	train-error:0+0	test-error:0.0124022+0.00400378
    [135]	train-error:0+0	test-error:0.0124022+0.00400378
    [136]	train-error:0+0	test-error:0.0124022+0.00400378
    [137]	train-error:0+0	test-error:0.0124022+0.00400378
    [138]	train-error:0+0	test-error:0.0124022+0.00400378
    [139]	train-error:0+0	test-error:0.0124022+0.00400378
    [140]	train-error:0+0	test-error:0.0124022+0.00400378
    [141]	train-error:0+0	test-error:0.0124022+0.00400378
    [142]	train-error:0+0	test-error:0.0124022+0.00400378
    [143]	train-error:0+0	test-error:0.0124022+0.00400378
    [144]	train-error:0+0	test-error:0.0120204+0.00398532
    [145]	train-error:0+0	test-error:0.0120204+0.00398532
    [146]	train-error:0+0	test-error:0.0120204+0.00398532
    [147]	train-error:0+0	test-error:0.0124022+0.00400378
    [148]	train-error:0+0	test-error:0.0124022+0.00400378
    [149]	train-error:0+0	test-error:0.0124022+0.00400378
    [150]	train-error:0+0	test-error:0.0124022+0.00400378
    [151]	train-error:0+0	test-error:0.0124022+0.00400378
    [152]	train-error:0+0	test-error:0.0124022+0.00400378
    [153]	train-error:0+0	test-error:0.0124022+0.00400378
    [154]	train-error:0+0	test-error:0.0124022+0.00400378
    [155]	train-error:0+0	test-error:0.0124022+0.00400378
    [156]	train-error:0+0	test-error:0.0124022+0.00400378
    [157]	train-error:0+0	test-error:0.0124022+0.00400378
    [158]	train-error:0+0	test-error:0.0124022+0.00400378
    [159]	train-error:0+0	test-error:0.0124022+0.00400378
    [160]	train-error:0+0	test-error:0.0124022+0.00400378
    [161]	train-error:0+0	test-error:0.0124022+0.00400378
    [162]	train-error:0+0	test-error:0.0124022+0.00400378
    [163]	train-error:0+0	test-error:0.0124022+0.00400378
    [164]	train-error:0+0	test-error:0.0124022+0.00400378
    [165]	train-error:0+0	test-error:0.0124022+0.00400378
    [166]	train-error:0+0	test-error:0.0124022+0.00400378
    [167]	train-error:0+0	test-error:0.0124022+0.00400378
    [168]	train-error:0+0	test-error:0.0124022+0.00400378
    [169]	train-error:0+0	test-error:0.0124022+0.00400378
    [170]	train-error:0+0	test-error:0.0124022+0.00400378
    [171]	train-error:0+0	test-error:0.0124022+0.00400378
    [172]	train-error:0+0	test-error:0.0124022+0.00400378
    [173]	train-error:0+0	test-error:0.0124022+0.00400378
    [174]	train-error:0+0	test-error:0.0122114+0.00369136
    [175]	train-error:0+0	test-error:0.0122114+0.00369136
    [176]	train-error:0+0	test-error:0.0122114+0.00369136
    [177]	train-error:0+0	test-error:0.0122114+0.00369136
    [178]	train-error:0+0	test-error:0.0122114+0.00369136
    [179]	train-error:0+0	test-error:0.0122114+0.00369136
    [180]	train-error:0+0	test-error:0.0122114+0.00369136
    [181]	train-error:0+0	test-error:0.0122114+0.00369136
    [182]	train-error:0+0	test-error:0.0122114+0.00369136
    [183]	train-error:0+0	test-error:0.0122114+0.00369136
    [184]	train-error:0+0	test-error:0.0122114+0.00369136
    [185]	train-error:0+0	test-error:0.0122114+0.00369136
    [186]	train-error:0+0	test-error:0.0122114+0.00369136
    [187]	train-error:0+0	test-error:0.0122114+0.00369136
    [188]	train-error:0+0	test-error:0.0122114+0.00369136
    [189]	train-error:0+0	test-error:0.0122114+0.00369136
    [190]	train-error:0+0	test-error:0.0122114+0.00369136
    [191]	train-error:0+0	test-error:0.0122114+0.00369136
    [192]	train-error:0+0	test-error:0.0122114+0.00369136
    [193]	train-error:0+0	test-error:0.0122114+0.00369136
    [194]	train-error:0+0	test-error:0.0122114+0.00369136





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>test-error-mean</th>
      <th>test-error-std</th>
      <th>train-error-mean</th>
      <th>train-error-std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.077464</td>
      <td>0.009777</td>
      <td>0.054379</td>
      <td>0.005647</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.074792</td>
      <td>0.011266</td>
      <td>0.047653</td>
      <td>0.001964</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.067542</td>
      <td>0.011053</td>
      <td>0.043122</td>
      <td>0.002693</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.066206</td>
      <td>0.010479</td>
      <td>0.039449</td>
      <td>0.001234</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.065824</td>
      <td>0.010621</td>
      <td>0.035489</td>
      <td>0.001017</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.060672</td>
      <td>0.010422</td>
      <td>0.032150</td>
      <td>0.001951</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.058002</td>
      <td>0.009180</td>
      <td>0.028811</td>
      <td>0.001592</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.054949</td>
      <td>0.009909</td>
      <td>0.026522</td>
      <td>0.002109</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.053041</td>
      <td>0.010040</td>
      <td>0.024089</td>
      <td>0.001100</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.050370</td>
      <td>0.009685</td>
      <td>0.021465</td>
      <td>0.000867</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.047126</td>
      <td>0.009186</td>
      <td>0.019367</td>
      <td>0.001993</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.044836</td>
      <td>0.008832</td>
      <td>0.016838</td>
      <td>0.001528</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.042356</td>
      <td>0.008806</td>
      <td>0.015503</td>
      <td>0.001972</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.041403</td>
      <td>0.007145</td>
      <td>0.014358</td>
      <td>0.001818</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.040067</td>
      <td>0.006431</td>
      <td>0.013213</td>
      <td>0.001543</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.037396</td>
      <td>0.007612</td>
      <td>0.011925</td>
      <td>0.001531</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.035679</td>
      <td>0.007664</td>
      <td>0.010208</td>
      <td>0.001578</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.033389</td>
      <td>0.008592</td>
      <td>0.009063</td>
      <td>0.001357</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.032435</td>
      <td>0.007931</td>
      <td>0.008252</td>
      <td>0.001371</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.031291</td>
      <td>0.008123</td>
      <td>0.007394</td>
      <td>0.001366</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.030337</td>
      <td>0.007897</td>
      <td>0.006869</td>
      <td>0.000859</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.029383</td>
      <td>0.007471</td>
      <td>0.005581</td>
      <td>0.000862</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.026712</td>
      <td>0.007461</td>
      <td>0.004627</td>
      <td>0.000862</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.024804</td>
      <td>0.008072</td>
      <td>0.003577</td>
      <td>0.000692</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.024040</td>
      <td>0.007616</td>
      <td>0.003387</td>
      <td>0.000745</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.022514</td>
      <td>0.007621</td>
      <td>0.002814</td>
      <td>0.000698</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.021560</td>
      <td>0.007717</td>
      <td>0.002433</td>
      <td>0.000552</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.019843</td>
      <td>0.006538</td>
      <td>0.002003</td>
      <td>0.000467</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.018508</td>
      <td>0.006172</td>
      <td>0.001574</td>
      <td>0.000387</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.018317</td>
      <td>0.006165</td>
      <td>0.001383</td>
      <td>0.000350</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>115</th>
      <td>0.012211</td>
      <td>0.004327</td>
      <td>0.000048</td>
      <td>0.000095</td>
    </tr>
    <tr>
      <th>116</th>
      <td>0.012211</td>
      <td>0.004327</td>
      <td>0.000048</td>
      <td>0.000095</td>
    </tr>
    <tr>
      <th>117</th>
      <td>0.012211</td>
      <td>0.004327</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>118</th>
      <td>0.012211</td>
      <td>0.004327</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>119</th>
      <td>0.012211</td>
      <td>0.004327</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>120</th>
      <td>0.012211</td>
      <td>0.004327</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>121</th>
      <td>0.012211</td>
      <td>0.004327</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>122</th>
      <td>0.012211</td>
      <td>0.004327</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>123</th>
      <td>0.012211</td>
      <td>0.004327</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>124</th>
      <td>0.012211</td>
      <td>0.004327</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>125</th>
      <td>0.012211</td>
      <td>0.004327</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>126</th>
      <td>0.012593</td>
      <td>0.004327</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>127</th>
      <td>0.012593</td>
      <td>0.004327</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>128</th>
      <td>0.012593</td>
      <td>0.004327</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>129</th>
      <td>0.012593</td>
      <td>0.004327</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>130</th>
      <td>0.012593</td>
      <td>0.004327</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>131</th>
      <td>0.012593</td>
      <td>0.004327</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>132</th>
      <td>0.012593</td>
      <td>0.004327</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>133</th>
      <td>0.012593</td>
      <td>0.004327</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>134</th>
      <td>0.012402</td>
      <td>0.004004</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>135</th>
      <td>0.012402</td>
      <td>0.004004</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>136</th>
      <td>0.012402</td>
      <td>0.004004</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>137</th>
      <td>0.012402</td>
      <td>0.004004</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>138</th>
      <td>0.012402</td>
      <td>0.004004</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>139</th>
      <td>0.012402</td>
      <td>0.004004</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>140</th>
      <td>0.012402</td>
      <td>0.004004</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>141</th>
      <td>0.012402</td>
      <td>0.004004</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>142</th>
      <td>0.012402</td>
      <td>0.004004</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>143</th>
      <td>0.012402</td>
      <td>0.004004</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>144</th>
      <td>0.012020</td>
      <td>0.003985</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>145 rows × 4 columns</p>
</div>





```python
param['num_parallel_tree'] = 50
num_round = 14
bst = xgb.train(param.items(),dtrain,num_round, evallist, early_stopping_rounds=5)
print (bst)
```


    [0]	eval-error:0.124204	train-error:0.04999
    Multiple eval metrics have been passed: 'train-error' will be used for early stopping.
    
    Will train until train-error hasn't improved in 5 rounds.
    [1]	eval-error:0.125796	train-error:0.042358
    [2]	eval-error:0.122611	train-error:0.041595
    [3]	eval-error:0.128981	train-error:0.038351
    [4]	eval-error:0.128981	train-error:0.034535
    [5]	eval-error:0.125796	train-error:0.032627
    [6]	eval-error:0.124204	train-error:0.028239
    [7]	eval-error:0.122611	train-error:0.024614
    [8]	eval-error:0.116242	train-error:0.022515
    [9]	eval-error:0.119427	train-error:0.019462
    [10]	eval-error:0.119427	train-error:0.017745
    [11]	eval-error:0.119427	train-error:0.016218
    [12]	eval-error:0.116242	train-error:0.015264
    [13]	eval-error:0.11465	train-error:0.013547
    <xgboost.core.Booster object at 0x000001D54C1B42E8>




```python
y_pred = bst.predict(dtest)
```




```python
df_fpr = pd.DataFrame(np.transpose([fpr,tpr]),columns=['fpr','tpr'])
df_fpr.to_csv('data/XGBoost_ROC_92.csv',index=False)
```


### Logistic Regression



```python
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
ll = []
i = 10**-5
while(i < 100):
    ll.append(i)
    i = i*5
logit = LogisticRegressionCV(cv=10,penalty='l2',Cs=ll)
logit.fit(x_train, y_train)
train_scores = logit.score(X_train_stand, y_train)
test_scores = logit.score(X_test_stand, y_test)
```




```python
fpr_log, tpr_log, thres_log = roc_curve(y_test, logit.predict_proba(X_test_stand)[:,1])
df_log = pd.DataFrame(np.transpose([fpr_log,tpr_log]),columns=['fpr','tpr'])
df_log.to_csv('data/LogReg_ROC_94.csv',index=False)
```


### ROC plot



```python
boosting_roc = pd.read_csv('./data/XGBoost_ROC_92.csv')
log_roc = pd.read_csv('./data/LogReg_ROC_94.csv')
fpr_b = boosting_roc.iloc[:,0]
tpr_b = boosting_roc.iloc[:,1]
fpr_l = log_roc.iloc[:,0]
tpr_l = log_roc.iloc[:,1]
```




```python
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("white")
sns.set_context("notebook")
```




```python
from sklearn.metrics import roc_curve, auc
def make_roc(name, fpr,tpr, ax=None, labe=5, proba=True, skip=0):
    initial=False
    if not ax:
        ax=plt.gca()
        initial=True
    roc_auc = auc(fpr, tpr)
    if skip:
        l=fpr.shape[0]
        ax.plot(fpr[0:l:skip], tpr[0:l:skip], '.-', alpha=1, label='ROC curve for %s (area = %0.2f)' % (name, roc_auc))
    else:
        ax.plot(fpr, tpr, '.-', alpha=0.5, label='ROC curve for %s (area = %0.2f)' % (name, roc_auc))

    if initial:
        ax.plot([-0.01, 1], [0, 1], 'k--',alpha=0.2)
        ax.set_xlim([-0.01, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC')
    ax.legend(loc="lower right")
    return ax, fpr, tpr, thresholds

```




```python
fig, ax = plt.subplots(1,figsize=(10,8))
ax, fpr, tpr, thresholds = make_roc("random forest", fpr1, tpr1, labe=10, skip=3)
ax, fpr, tpr, thresholds = make_roc("logistic regression", fpr_l, tpr_l, labe=10, skip=3)
ax, fpr, tpr, thresholds = make_roc("XGBoost", fpr_b, tpr_b, labe=10, skip=3)

```





    <matplotlib.legend.Legend at 0x11b06d9e8>




![png](models_files/models_51_1.png)


### Multinomial Regression



```python
data = pd.read_csv("ADNIMerge_PostImputation.csv")
```




```python
adni1 = data[data['COLPROT'] =='ADNI1']

columns = ['Unnamed: 0','VISCODE','EXAMDATE_bl', 'DX_bl','ORIGPROT','COLPROT','EXAMDATE','FAQ','FAQ_bl','Years_bl','Month_bl','M','PTETHCAT',
           'CDRSB','CDRSB_bl','MMSE','MMSE_bl','FLDSTRENG_bl','FSVERSION_bl','update_stamp','FLDSTRENG','FSVERSION','SITE','PTID',
          'RAVLT_immediate','RAVLT_learning','RAVLT_forgetting','RAVLT_perc_forgetting']
adni1 =adni1.drop(columns, axis=1)
adni1.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RID</th>
      <th>AGE</th>
      <th>PTGENDER</th>
      <th>PTEDUCAT</th>
      <th>PTRACCAT</th>
      <th>PTMARRY</th>
      <th>APOE4</th>
      <th>FDG</th>
      <th>PIB</th>
      <th>AV45</th>
      <th>...</th>
      <th>EcogSPLang_bl</th>
      <th>EcogSPVisspat_bl</th>
      <th>EcogSPPlan_bl</th>
      <th>EcogSPOrgan_bl</th>
      <th>EcogSPDivatt_bl</th>
      <th>EcogSPTotal_bl</th>
      <th>FDG_bl</th>
      <th>PIB_bl</th>
      <th>AV45_bl</th>
      <th>Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.537838</td>
      <td>Male</td>
      <td>16</td>
      <td>White</td>
      <td>Married</td>
      <td>0.0</td>
      <td>0.656019</td>
      <td>0.070941</td>
      <td>0.094271</td>
      <td>...</td>
      <td>0.555557</td>
      <td>0.095237</td>
      <td>0.333333</td>
      <td>0.944443</td>
      <td>0.666667</td>
      <td>0.531305</td>
      <td>0.665405</td>
      <td>0.365854</td>
      <td>0.237378</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0.727027</td>
      <td>Male</td>
      <td>18</td>
      <td>White</td>
      <td>Married</td>
      <td>0.5</td>
      <td>0.406609</td>
      <td>0.648022</td>
      <td>0.308011</td>
      <td>...</td>
      <td>0.259260</td>
      <td>0.047620</td>
      <td>0.200000</td>
      <td>0.222223</td>
      <td>0.500000</td>
      <td>0.249885</td>
      <td>0.389666</td>
      <td>0.654102</td>
      <td>0.415698</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.727027</td>
      <td>Male</td>
      <td>18</td>
      <td>White</td>
      <td>Married</td>
      <td>0.5</td>
      <td>0.382257</td>
      <td>0.552524</td>
      <td>0.344773</td>
      <td>...</td>
      <td>0.222223</td>
      <td>0.333333</td>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>0.250000</td>
      <td>0.286955</td>
      <td>0.389666</td>
      <td>0.944568</td>
      <td>0.372792</td>
      <td>0.045455</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.727027</td>
      <td>Male</td>
      <td>18</td>
      <td>White</td>
      <td>Married</td>
      <td>0.5</td>
      <td>0.418298</td>
      <td>0.477490</td>
      <td>0.369602</td>
      <td>...</td>
      <td>0.333333</td>
      <td>0.190477</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.226088</td>
      <td>0.389666</td>
      <td>0.585366</td>
      <td>0.451948</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0.727027</td>
      <td>Male</td>
      <td>18</td>
      <td>White</td>
      <td>Married</td>
      <td>0.5</td>
      <td>0.359964</td>
      <td>0.451569</td>
      <td>0.398939</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.095237</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.029067</td>
      <td>0.389666</td>
      <td>0.851441</td>
      <td>0.540995</td>
      <td>0.181818</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 67 columns</p>
</div>





```python
categorical_columns = ['PTGENDER','PTRACCAT', 'PTMARRY']

head = adni1.columns.values.tolist()
numerical_columns = [a for a in head if a not in categorical_columns]
numerical_columns = [x for x in numerical_columns if x != 'DX']
```




```python
adni1 = pd.get_dummies(adni1, columns=categorical_columns, drop_first=True)
adni1.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RID</th>
      <th>AGE</th>
      <th>PTEDUCAT</th>
      <th>APOE4</th>
      <th>FDG</th>
      <th>PIB</th>
      <th>AV45</th>
      <th>ADAS11</th>
      <th>ADAS13</th>
      <th>MOCA</th>
      <th>...</th>
      <th>Month</th>
      <th>PTGENDER_Male</th>
      <th>PTRACCAT_Asian</th>
      <th>PTRACCAT_Black</th>
      <th>PTRACCAT_More than one</th>
      <th>PTRACCAT_White</th>
      <th>PTMARRY_Married</th>
      <th>PTMARRY_Never married</th>
      <th>PTMARRY_Unknown</th>
      <th>PTMARRY_Widowed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.537838</td>
      <td>16</td>
      <td>0.0</td>
      <td>0.656019</td>
      <td>0.070941</td>
      <td>0.094271</td>
      <td>0.152429</td>
      <td>0.219647</td>
      <td>0.766667</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0.727027</td>
      <td>18</td>
      <td>0.5</td>
      <td>0.406609</td>
      <td>0.648022</td>
      <td>0.308011</td>
      <td>0.314286</td>
      <td>0.364706</td>
      <td>0.600000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.727027</td>
      <td>18</td>
      <td>0.5</td>
      <td>0.382257</td>
      <td>0.552524</td>
      <td>0.344773</td>
      <td>0.271429</td>
      <td>0.352941</td>
      <td>0.633333</td>
      <td>...</td>
      <td>0.045455</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.727027</td>
      <td>18</td>
      <td>0.5</td>
      <td>0.418298</td>
      <td>0.477490</td>
      <td>0.369602</td>
      <td>0.342857</td>
      <td>0.411765</td>
      <td>0.500000</td>
      <td>...</td>
      <td>0.090909</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0.727027</td>
      <td>18</td>
      <td>0.5</td>
      <td>0.359964</td>
      <td>0.451569</td>
      <td>0.398939</td>
      <td>0.366714</td>
      <td>0.443176</td>
      <td>0.500000</td>
      <td>...</td>
      <td>0.181818</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 73 columns</p>
</div>





```python
adni1.DX = pd.Categorical(adni1.DX).codes
adni1["DX"].values
```





    array([1, 2, 2, ..., 2, 0, 2], dtype=int8)





```python
gb = adni1.groupby('RID')    
groups=[gb.get_group(x) for x in gb.groups]


index_train = random.sample(range(0,len(groups)), int(len(groups)/2))
index_test = list(set(list(range(0,len(groups)))) - set(index_train))


train_df = pd.concat([groups[i] for i in index_train])
test_df = pd.concat([groups[i] for i in index_test])
```




```python
mean = train_df[numerical_columns].mean()
std = train_df[numerical_columns].std()

mean_t = test_df[numerical_columns].mean()
std_t = test_df[numerical_columns].std()

train_df[numerical_columns] = (train_df[numerical_columns] - mean)/std
test_df[numerical_columns] = (test_df[numerical_columns] - mean_t)/std_t
```




```python
X_train=train_df.drop(['DX','RID'], axis=1)
y_train=train_df['DX']

X_test=test_df.drop(['DX','RID'], axis=1)
y_test=test_df['DX']
```




```python
#Multinomial
multinomial = LogisticRegressionCV(multi_class = 'multinomial', solver = 'newton-cg',cv=5, penalty='l2')
multinomial.fit(X_train, y_train)

#One v. Rest
ovr = LogisticRegressionCV(multi_class = 'ovr')
ovr.fit(X_train, y_train)
```





    LogisticRegressionCV(Cs=10, class_weight=None, cv=None, dual=False,
               fit_intercept=True, intercept_scaling=1.0, max_iter=100,
               multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
               refit=True, scoring=None, solver='lbfgs', tol=0.0001, verbose=0)





```python
#Computing the score
print('OVR Logistic Regression Train Score: ',ovr.score(X_train, y_train))
print('Multinomial Logistic Regression Train Score: ',multinomial.score(X_train, y_train))

#Computing the score
print('OVR Logistic Regression Test Score: ',ovr.score(X_test, y_test))
print('Multinomial Logistic Regression Test Score: ',multinomial.score(X_test, y_test))
```


    OVR Logistic Regression Train Score:  0.667186890363
    Multinomial Logistic Regression Train Score:  0.671478735856
    OVR Logistic Regression Test Score:  0.609387755102
    Multinomial Logistic Regression Test Score:  0.611836734694

