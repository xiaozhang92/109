---
title: missing rate > 0.5
notebook: EDA.ipynb
nav_include: 1
---

## Contents
{:.no_toc}
*  
{: toc}



```python
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.api import OLS
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
%matplotlib inline
```


    /opt/anaconda3/lib/python3.6/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools


## EDA



```python
df_merge = pd.read_csv('./data/ADNIMERGE.csv',parse_dates = ['EXAMDATE','EXAMDATE_bl'])
```




```python
df_merge.head()
```





<div>
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
      <th>EcogSPDivatt_bl</th>
      <th>EcogSPTotal_bl</th>
      <th>FDG_bl</th>
      <th>PIB_bl</th>
      <th>AV45_bl</th>
      <th>Years_bl</th>
      <th>Month_bl</th>
      <th>Month</th>
      <th>M</th>
      <th>update_stamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>011_S_0002</td>
      <td>bl</td>
      <td>11</td>
      <td>ADNI1</td>
      <td>ADNI1</td>
      <td>2005-09-08</td>
      <td>CN</td>
      <td>74.3</td>
      <td>Male</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.36926</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0</td>
      <td>0</td>
      <td>2017-08-13 23:50:48.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>011_S_0003</td>
      <td>bl</td>
      <td>11</td>
      <td>ADNI1</td>
      <td>ADNI1</td>
      <td>2005-09-12</td>
      <td>AD</td>
      <td>81.3</td>
      <td>Male</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.09079</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0</td>
      <td>0</td>
      <td>2017-08-13 23:50:48.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>011_S_0003</td>
      <td>m06</td>
      <td>11</td>
      <td>ADNI1</td>
      <td>ADNI1</td>
      <td>2006-03-13</td>
      <td>AD</td>
      <td>81.3</td>
      <td>Male</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.09079</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.498289</td>
      <td>5.96721</td>
      <td>6</td>
      <td>6</td>
      <td>2017-08-13 23:50:48.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>011_S_0003</td>
      <td>m12</td>
      <td>11</td>
      <td>ADNI1</td>
      <td>ADNI1</td>
      <td>2006-09-12</td>
      <td>AD</td>
      <td>81.3</td>
      <td>Male</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.09079</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.999316</td>
      <td>11.96720</td>
      <td>12</td>
      <td>12</td>
      <td>2017-08-13 23:50:48.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>011_S_0003</td>
      <td>m24</td>
      <td>11</td>
      <td>ADNI1</td>
      <td>ADNI1</td>
      <td>2007-09-12</td>
      <td>AD</td>
      <td>81.3</td>
      <td>Male</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.09079</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.998630</td>
      <td>23.93440</td>
      <td>24</td>
      <td>24</td>
      <td>2017-08-13 23:50:48.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 94 columns</p>
</div>





```python
df_merge.columns
```





    Index(['RID', 'PTID', 'VISCODE', 'SITE', 'COLPROT', 'ORIGPROT', 'EXAMDATE',
           'DX_bl', 'AGE', 'PTGENDER', 'PTEDUCAT', 'PTETHCAT', 'PTRACCAT',
           'PTMARRY', 'APOE4', 'FDG', 'PIB', 'AV45', 'CDRSB', 'ADAS11', 'ADAS13',
           'MMSE', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting',
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
           'update_stamp'],
          dtype='object')





```python
DX = set(df_merge['DX'].dropna())
DX
```





    {'CN', 'Dementia', 'MCI'}





```python
PROJ = set(df_merge['COLPROT'])
PROJ
```





    {'ADNI1', 'ADNI2', 'ADNI3', 'ADNIGO'}





```python
### APOE4 ###
df_merge.groupby('DX').APOE4.mean()
```





    DX
    CN          0.292168
    Dementia    0.865147
    MCI         0.549716
    Name: APOE4, dtype: float64





```python
### BETA-AMYLOID ###

df_merge['PIB'].hist()
plt.title('Histogram of PIB')
plt.xlim(0.75,3)
```





    (0.75, 3)




![png](EDA_files/EDA_8_1.png)




```python
df_merge['AV45'].hist()
plt.title('Histogram of AV45')
plt.xlim(0.75, 3)
```





    (0.75, 3)




![png](EDA_files/EDA_9_1.png)




```python
df_merge[['PIB','AV45']].describe()
```





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PIB</th>
      <th>AV45</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>223.000000</td>
      <td>2161.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.783161</td>
      <td>1.195504</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.422511</td>
      <td>0.227999</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.095000</td>
      <td>0.814555</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.361250</td>
      <td>1.010140</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.850000</td>
      <td>1.114670</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.127500</td>
      <td>1.364980</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.927500</td>
      <td>2.669210</td>
    </tr>
  </tbody>
</table>
</div>





```python
for p in ['ADNI1', 'ADNI2', 'ADNIGO']:
    print(p)
    if p == 'ADNI1':
        beta_col = ['PIB']
    else:
        beta_col = ['AV45']

    df_merge_sub = df_merge.loc[df_merge['COLPROT']==p,:]
    fig, ax = plt.subplots()
    
    for col in beta_col:
        for d in DX:
            ax = df_merge_sub.loc[df_merge_sub['DX']==d, col].plot(kind = 'density', label=d)
        plt.legend()
        plt.title('Density Plot of ' + col +" in " + p)
        plt.show()
```


    ADNI1



![png](EDA_files/EDA_11_1.png)


    ADNI2



![png](EDA_files/EDA_11_3.png)


    ADNIGO



![png](EDA_files/EDA_11_5.png)




```python
### FDG ###
for p in ['ADNI1', 'ADNI2', 'ADNIGO']:
    print(p)
    df_merge_sub = df_merge.loc[df_merge['COLPROT']==p,:]
    fig, ax = plt.subplots()

    for d in DX:
        ax = df_merge_sub.loc[df_merge_sub['DX']==d, 'FDG'].plot(kind = 'density', label=d)
    plt.legend()
    plt.title('Density Plot of FDG'+" in " + p)
    plt.show()
```


    ADNI1



![png](EDA_files/EDA_12_1.png)


    ADNI2



![png](EDA_files/EDA_12_3.png)


    ADNIGO



![png](EDA_files/EDA_12_5.png)




```python
len(set(df_merge['RID']))
```





    1784





```python
### COGNITIVE CRITERIA ###
df_merge_cog = df_merge[['CDRSB','MMSE','ADAS11','ADAS13']].dropna(axis = 0, how = 'any')
df_merge_cog.head()
```





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CDRSB</th>
      <th>MMSE</th>
      <th>ADAS11</th>
      <th>ADAS13</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>28.0</td>
      <td>10.67</td>
      <td>18.67</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.5</td>
      <td>20.0</td>
      <td>22.00</td>
      <td>31.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.0</td>
      <td>24.0</td>
      <td>19.00</td>
      <td>30.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.5</td>
      <td>17.0</td>
      <td>24.00</td>
      <td>35.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.0</td>
      <td>19.0</td>
      <td>25.67</td>
      <td>37.67</td>
    </tr>
  </tbody>
</table>
</div>





```python
corr_mat = np.corrcoef(df_merge_cog.T)
fig, ax = plt.subplots(figsize=(8,6))
plt.pcolor(corr_mat)
ax.set_xticks([float(x)+0.5 for x in range(len(df_merge_cog.columns))])
ax.set_xticklabels(df_merge_cog.columns,rotation=90)
ax.set_yticks([float(x)+0.5 for x in range(len(df_merge_cog.columns))])
ax.set_yticklabels(df_merge_cog.columns)
plt.title('Correlation between Cognitive Tests')
plt.colorbar()
```





    <matplotlib.colorbar.Colorbar at 0x1e62f0966a0>




![png](EDA_files/EDA_15_1.png)




```python
pd.DataFrame(corr_mat, columns=df_merge_cog.columns, index=df_merge_cog.columns)
```





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CDRSB</th>
      <th>MMSE</th>
      <th>ADAS11</th>
      <th>ADAS13</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CDRSB</th>
      <td>1.000000</td>
      <td>-0.791662</td>
      <td>0.795000</td>
      <td>0.801335</td>
    </tr>
    <tr>
      <th>MMSE</th>
      <td>-0.791662</td>
      <td>1.000000</td>
      <td>-0.837666</td>
      <td>-0.838212</td>
    </tr>
    <tr>
      <th>ADAS11</th>
      <td>0.795000</td>
      <td>-0.837666</td>
      <td>1.000000</td>
      <td>0.981229</td>
    </tr>
    <tr>
      <th>ADAS13</th>
      <td>0.801335</td>
      <td>-0.838212</td>
      <td>0.981229</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>





```python
### MMSE
for p in PROJ:
    print(p)
    df_merge_sub = df_merge.loc[df_merge['COLPROT']==p,:]
    fig, ax = plt.subplots()

    for d in DX:
        ax = df_merge_sub.loc[df_merge_sub['DX']==d, 'MMSE'].plot(kind = 'density', label=d)
    plt.legend()
    plt.title('Density Plot of MMSE'+" in " + p)
    plt.show()
```


    ADNI1



![png](EDA_files/EDA_17_1.png)


    ADNIGO



![png](EDA_files/EDA_17_3.png)


    ADNI2



![png](EDA_files/EDA_17_5.png)


    ADNI3



![png](EDA_files/EDA_17_7.png)




```python
### ADAS11
for p in PROJ:
    print(p)
    df_merge_sub = df_merge.loc[df_merge['COLPROT']==p,:]
    fig, ax = plt.subplots()

    for d in DX:
        ax = df_merge_sub.loc[df_merge_sub['DX']==d, 'ADAS11'].plot(kind = 'density', label=d)
    plt.legend()
    plt.title('Density Plot of ADAS11'+" in " + p)
    plt.show()
```


    ADNI1



![png](EDA_files/EDA_18_1.png)


    ADNIGO



![png](EDA_files/EDA_18_3.png)


    ADNI2



![png](EDA_files/EDA_18_5.png)


    ADNI3



![png](EDA_files/EDA_18_7.png)


## Missing Data



```python
data = pd.read_csv('./data/ADNIMERGE.csv',parse_dates = ['EXAMDATE','EXAMDATE_bl'])
```




```python
head = data.columns.values.tolist()

miss_rate=[]
for i in range(0,94):
    miss = np.count_nonzero(data.iloc[:,i].isnull().values)
    rate = miss/13017
    miss_rate.append(float(rate))
```




```python
new_df = pd.DataFrame(np.column_stack([head, miss_rate]), 
                               columns=['head', 'miss_rate'])

new_df['miss_rate'] = pd.to_numeric(new_df['miss_rate'], errors='coerce')
```




```python
miss_all = new_df[new_df['miss_rate'] >0]

miss_all["head"].values
```





    array(['DX_bl', 'APOE4', 'FDG', 'PIB', 'AV45', 'CDRSB', 'ADAS11', 'ADAS13',
           'MMSE', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting',
           'RAVLT_perc_forgetting', 'FAQ', 'MOCA', 'EcogPtMem', 'EcogPtLang',
           'EcogPtVisspat', 'EcogPtPlan', 'EcogPtOrgan', 'EcogPtDivatt',
           'EcogPtTotal', 'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat',
           'EcogSPPlan', 'EcogSPOrgan', 'EcogSPDivatt', 'EcogSPTotal',
           'FLDSTRENG', 'FSVERSION', 'Ventricles', 'Hippocampus', 'WholeBrain',
           'Entorhinal', 'Fusiform', 'MidTemp', 'ICV', 'DX', 'ADAS11_bl',
           'ADAS13_bl', 'RAVLT_immediate_bl', 'RAVLT_learning_bl',
           'RAVLT_forgetting_bl', 'RAVLT_perc_forgetting_bl', 'FAQ_bl',
           'FLDSTRENG_bl', 'FSVERSION_bl', 'Ventricles_bl', 'Hippocampus_bl',
           'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl',
           'ICV_bl', 'MOCA_bl', 'EcogPtMem_bl', 'EcogPtLang_bl',
           'EcogPtVisspat_bl', 'EcogPtPlan_bl', 'EcogPtOrgan_bl',
           'EcogPtDivatt_bl', 'EcogPtTotal_bl', 'EcogSPMem_bl',
           'EcogSPLang_bl', 'EcogSPVisspat_bl', 'EcogSPPlan_bl',
           'EcogSPOrgan_bl', 'EcogSPDivatt_bl', 'EcogSPTotal_bl', 'FDG_bl',
           'PIB_bl', 'AV45_bl'], dtype=object)





```python
miss = new_df[new_df['miss_rate'] >0.5]

miss["head"].values
```





    array(['FDG', 'PIB', 'AV45', 'MOCA', 'EcogPtMem', 'EcogPtLang',
           'EcogPtVisspat', 'EcogPtPlan', 'EcogPtOrgan', 'EcogPtDivatt',
           'EcogPtTotal', 'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat',
           'EcogSPPlan', 'EcogSPOrgan', 'EcogSPDivatt', 'EcogSPTotal',
           'Entorhinal', 'Fusiform', 'MidTemp', 'MOCA_bl', 'EcogPtMem_bl',
           'EcogPtLang_bl', 'EcogPtVisspat_bl', 'EcogPtPlan_bl',
           'EcogPtOrgan_bl', 'EcogPtDivatt_bl', 'EcogPtTotal_bl',
           'EcogSPMem_bl', 'EcogSPLang_bl', 'EcogSPVisspat_bl',
           'EcogSPPlan_bl', 'EcogSPOrgan_bl', 'EcogSPDivatt_bl',
           'EcogSPTotal_bl', 'PIB_bl', 'AV45_bl'], dtype=object)


