# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 18:08:45 2021

@author: User

"""

import numpy as np 
import matplotlib.pyplot as plt #畫出圖型 
import pandas as pd #資料處理
from imblearn.over_sampling import SMOTE

# Importing the dataset 

df = pd.read_excel("tr.xlsx", engine='openpyxl')
print(df)

print("==================================================")

#==============================================================#

# 顯示所有列
pd.set_option('display.max_columns', None)
# 顯示所有行
pd.set_option('display.max_rows', None)
# 設置value的顯示長度為100，默認為50
pd.set_option('max_colwidth',100)


# 處理缺失數據
# 判斷各變數中是否存在缺失值
print(df.isnull().any(axis = 0))
print("==================================================")

# 各變數中缺失值的數量
print(df.isnull().sum(axis = 0))
print("==================================================")

# 各變數中缺失值的比例
print(df.isnull().sum(axis = 0)/df.shape[0])
print("==================================================")

#==============================================================#
# 視覺化缺失值的分佈

# 統計缺失值數量
missing=df.isnull().sum().reset_index().rename(columns={0:'missNum'})
# 計算缺失比例
missing['missRate']=missing['missNum']/df.shape[0]
# 按照缺失率排序顯示
miss_analy=missing[missing.missRate>-1].sort_values(by='missRate',ascending=False)
# miss_analy 儲存的是每個變數缺失情況的資料框

fig = plt.figure(figsize=(20,20))
plt.bar(np.arange(miss_analy.shape[0]), list(miss_analy.missRate.values), align = 'center'
    ,color=['red','green','yellow','steelblue'])

plt.title('Histogram of missing value of variables')
plt.xlabel('variables names')
plt.ylabel('missing rate')
# 新增x軸標籤，並旋轉90度
plt.xticks(np.arange(miss_analy.shape[0]),list(miss_analy['index']))
plt.xticks(rotation=90)
# 新增數值顯示
for x,y in enumerate(list(miss_analy.missRate.values)):
    plt.text(x,y+0.12,'{:.2%}'.format(y),ha='center',rotation=90)    
plt.ylim([0,1.2])
    
plt.show()

#==============================================================#

# 用相關性探索缺失資料
missing_depend=df.isnull()
corr_matrix = missing_depend.corr()
#corr_matrix["missing_variable"].sort_values(ascending=False)
print(corr_matrix)
print("==================================================")

#==============================================================#

# 缺失資料的處理

# 隨機插值
# iterating the columns
_dict = {}
for col in df.columns:
    try:
        _dict[col] = df[col].mean() # tr
        # _dict[col] = ch[col] # ts
    except:
        print('e')
    print(col)

# .mode()[0] 眾數替换
# .mean() 平均值替换
df.fillna(value = _dict, inplace = True )

df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))

print("==================================================")

#==============================================================#

# 分類型數據轉化數值型

sex_mapping = {"M": 0, "F": 1}
df["SEX"] = df["SEX"].map(sex_mapping)

joint_mapping = {"TKA": 0, "THA": 1}
df["Joint"] = df["Joint"].map(joint_mapping)

print("==================================================")

#==============================================================#

# 數據格式轉換

# def mem_usage(pandas_obj):
#     if isinstance(pandas_obj,pd.DataFrame):
#         usage_b = pandas_obj.memory_usage(deep=True).sum()
#     else: # we assume if not a df it's a series
#         usage_b = pandas_obj.memory_usage(deep=True)
#     usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
#     return "{:03.2f} MB".format(usage_mb)

# # int
# gl_int = df.select_dtypes(include=['int64']).apply(pd.to_numeric,downcast='unsigned')
# converted_int = gl_int.apply(pd.to_numeric,downcast='unsigned')
# print("before: ", mem_usage(gl_int))
# print("after: ", mem_usage(converted_int))
# compare_ints = pd.concat([gl_int.dtypes,converted_int.dtypes],axis=1)

# # float
# gl_float = df.select_dtypes(include=['float'])
# converted_float = gl_float.apply(pd.to_numeric,downcast='float')
# print("before: ", mem_usage(gl_float))
# print("after: ", mem_usage(converted_float))
# compare_floats = pd.concat([gl_float.dtypes,converted_float.dtypes],axis=1)

# # object
# gl_obj = df.select_dtypes(include=['object']).copy()
# converted_obj = pd.DataFrame()

# for col in gl_obj.columns:
#     num_unique_values = len(gl_obj[col].unique())
#     num_total_values = len(gl_obj[col])
#     if num_unique_values / num_total_values < 0.5:
#         converted_obj.loc[:,col] = gl_obj[col].astype('category')
#     else:
#         converted_obj.loc[:,col] = gl_obj[col]

df['outcome'] = df['outcome'].astype('uintc') 
df['Drain'] = df['Drain'].astype('uintc') 
df['Cemented'] = df['Cemented'].astype('uintc') 
df['Commercial_ALBC'] = df['Commercial_ALBC'].astype('uintc') 
df['Non_commercial_ALBC'] = df['Non_commercial_ALBC'].astype('uintc') 
df['cci_index'] = df['cci_index'].astype('uintc') 
df['elx_index'] = df['elx_index'].astype('uintc') 
df['Blood_trans'] = df['Blood_trans'].astype('uintc') 
df['Diagnosis'] = df['Diagnosis'].astype('uintc') 
df['Congestive Heart Failure'] = df['Congestive Heart Failure'].astype('uintc') 
df['Cardiac Arrhythmia'] = df['Cardiac Arrhythmia'].astype('uintc') 
df['Valvular Disease'] = df['Valvular Disease'].astype('uintc') 
df['Heart disease'] = df['Heart disease'].astype('uintc') 
df['Pulmonary Circulation Disorders'] = df['Pulmonary Circulation Disorders'].astype('uintc') 
df['Peripheral Vascular Disorders'] = df['Peripheral Vascular Disorders'].astype('uintc') 
df['Hypertension Uncomplicated'] = df['Hypertension Uncomplicated'].astype('uintc') 
df['Paralysis'] = df['Paralysis'].astype('uintc') 
df['Other Neurological Disorders'] = df['Other Neurological Disorders'].astype('uintc') 
df['Chronic Pulmonary Disease'] = df['Chronic Pulmonary Disease'].astype('uintc') 
df['Lung disease'] = df['Lung disease'].astype('uintc') 
df['Diabetes'] = df['Diabetes'].astype('uintc') 
df['Hypothyroidism'] = df['Hypothyroidism'].astype('uintc') 
df['Renal Failure'] = df['Renal Failure'].astype('uintc') 
df['Liver Disease'] = df['Liver Disease'].astype('uintc') 
df['Peptic Ulcer Disease excluding bleeding'] = df['Peptic Ulcer Disease excluding bleeding'].astype('uintc') 
df['AIDS/HIV'] = df['AIDS/HIV'].astype('uintc') 
df['Lymphoma'] = df['Lymphoma'].astype('uintc') 
df['Metastatic Cancer'] = df['Metastatic Cancer'].astype('uintc') 
df['Solid Tumor without Metastasis'] = df['Solid Tumor without Metastasis'].astype('uintc') 
df['Cancer history'] = df['Cancer history'].astype('uintc') 
df['Rheumatoid Arthritis/collagen'] = df['Rheumatoid Arthritis/collagen'].astype('uintc') 
df['Coagulopathy'] = df['Coagulopathy'].astype('uintc') 
df['Obesity'] = df['Obesity'].astype('uintc') 
df['Weight Loss'] = df['Weight Loss'].astype('uintc') 
df['Fluid and Electrolyte Disorders'] = df['Fluid and Electrolyte Disorders'].astype('uintc') 
df['Blood Loss Anemia'] = df['Blood Loss Anemia'].astype('uintc') 
df['Deficiency Anemia'] = df['Deficiency Anemia'].astype('uintc') 
df['Anemia'] = df['Anemia'].astype('uintc') 
df['Alcohol Abuse'] = df['Alcohol Abuse'].astype('uintc') 
df['Drug Abuse'] = df['Drug Abuse'].astype('uintc') 
df['Psychoses'] = df['Psychoses'].astype('uintc') 
df['Depression'] = df['Depression'].astype('uintc') 
df['Psyciatric disorder'] = df['Psyciatric disorder'].astype('uintc') 
df['AGE'] = df['AGE'].astype('uintc') 
df['ASA'] = df['ASA'].astype('uintc') 
df['LOS'] = df['LOS'].astype('float') 
df['OP_time_minute'] = df['OP_time_minute'].astype('float') 
df['OP_time_hour'] = df['OP_time_hour'].astype('float') 
df['CBC_WBC'] = df['CBC_WBC'].astype('float') 
df['CBC_RBC'] = df['CBC_RBC'].astype('float') 
df['CBC_HG'] = df['CBC_HG'].astype('float') 
df['CBC_HT'] = df['CBC_HT'].astype('float') 
df['CBC_MCV'] = df['CBC_MCV'].astype('float') 
df['CBC_MCH'] = df['CBC_MCH'].astype('float') 
df['CBC_MCHC'] = df['CBC_MCHC'].astype('float') 
df['CBC_RDW'] = df['CBC_RDW'].astype('float') 
df['CBC_Platelet'] = df['CBC_Platelet'].astype('float') 
df['CBC_RDWCV'] = df['CBC_RDWCV'].astype('float') 
df['BUN'] = df['BUN'].astype('float') 
df['Crea'] = df['Crea'].astype('float') 
df['GOT'] = df['GOT'].astype('float') 
df['GPT'] = df['GPT'].astype('float') 
df['ALB'] = df['ALB'].astype('float') 
df['Na'] = df['Na'].astype('float') 
df['K'] = df['K'].astype('float') 
df['UA'] = df['UA'].astype('float') 
df['SEX'] = df['SEX'].astype('uintc') # category
df['Joint'] = df['Joint'].astype('uintc') # category



print("==================================================")

#==============================================================#


# 刪除相關性高的行

df = df.drop(['OP_time_minute'], axis=1)

print("==================================================")

#==============================================================#

# 刪除重複數據

df = df.drop_duplicates()

print("==================================================")

#==============================================================#

# 數據異常值

mean1 = df['GOT'].quantile(q=0.25)#下四分位差
mean2 = df['GOT'].quantile(q=0.75)#上四分位差
mean3 = mean2 - mean1#中位差
topnum2 = mean2 + 1.5 * mean3
bottomnum2 = mean2 - 1.5 * mean3
replace_value1 = df['GOT'][df['GOT'] < topnum2].max()
df.loc[df['GOT'] > topnum2,'GOT'] = replace_value1
replace_value2 = df['GOT'][df['GOT'] > bottomnum2].min()
df.loc[df['GOT'] < bottomnum2,'GOT'] = replace_value2

print("==================================================")

#==============================================================#

# 特徵標準化/歸一化

def mean_norm(df_input):
    df_input = (df_input-df_input.mean())/ df_input.std()
    return df_input

df["LOS"] = mean_norm(df["LOS"])

print("==================================================")


#==============================================================#

# 刪除異常值

df = df.loc[df["Cancer history"] < 3]

#==============================================================#

# 四捨五入到自訂的小數位數

df = df.round(decimals=2)  #四捨五入到整數位
df['AGE'] = df['AGE'].round(decimals=0)  #四捨五入到整數位
df['ASA'] = df['ASA'].round(decimals=0)  #四捨五入到整數位

print("==================================================")

#==============================================================#

# 移動col位置
# shift column to first position
first_column = df.pop('outcome')
  
# insert last column
df['outcome'] =  first_column

# print(df)

print("==================================================")

# 再次處理缺失數據
# 判斷各變數中是否存在缺失值
print(df.isnull().any(axis = 0))
print("==================================================")

# 各變數中缺失值的數量
print(df.isnull().sum(axis = 0))
print("==================================================")

# Separate majority and minority classes
df_majority = df[df.outcome==0]
df_minority = df[df.outcome==1]

print(len(df))
print(len(df_majority))
print(len(df_minority))

# 數據不平衡
x, y = df.iloc[:,:-1], df.iloc[:,-1]

smote_model = SMOTE()
x_smote, y_smote = smote_model.fit_sample(x, y)
df = pd.concat([x_smote, y_smote], axis = 1)

# Separate majority and minority classes
df_majority = df[df.outcome==0]
df_minority = df[df.outcome==1]

print(len(df))
print(len(df_majority))
print(len(df_minority))



print("==================================================")

# 四捨五入到自訂的小數位數

df = df.round(decimals=2)  #四捨五入到整數位
df['AGE'] = df['AGE'].round(decimals=0)  #四捨五入到整數位
df['ASA'] = df['ASA'].round(decimals=0)  #四捨五入到整數位

print("==================================================")

#==============================================================#

# 隨機插值
# iterating the columns
_dict = {}
for col in df.columns:
    try:
        _dict[col] = df[col].mean() # tr
        # _dict[col] = ch[col] # ts
    except:
        print('e')
    print(col)

# .mode()[0] 眾數替换
# .mean() 平均值替换
df.fillna(value = _dict, inplace = True )

df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))

print("==================================================")

# 輸出excel
#df.to_excel('output.xlsx', index=False)

# 輸出csv
df.to_csv("output_final6.csv", index=False)

# # 按比例隨機採樣
# # 按100%的比例抽樣即達到打亂數據的效果
# df = df.sample(frac = 1.0) 

# # Creating a dataframe with 50%
# # values of original dataframe
# test = df.sample(frac = 0.25)
# # Creating dataframe with
# # rest of the 50% values
# train = df.drop(test.index)

# print(len(train))
# print(len(test))

# # 輸出csv
# train.to_csv("output_train.csv", index=False)

# # 輸出csv
# test.to_csv("output_test.csv", index=False)