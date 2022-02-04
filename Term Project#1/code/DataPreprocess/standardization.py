# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 03:10:39 2021

@author: User
"""

import numpy as np 
import matplotlib.pyplot as plt #畫出圖型 
import pandas as pd #資料處理

# Importing the dataset 

df = pd.read_excel("tr.xlsx", engine='openpyxl')


# 空白插值
# iterating the columns
_dict = {}
for col in df.columns:
    try:
        _dict[col] = df[col].mean()
    except:
        print('e')
    print(col)

# .mode()[0] 眾數替换
# .mean() 平均值替换
df.fillna(value = _dict, inplace = True )

# 標準/歸一化
def mean_norm(df_input):
    df_input = (df_input-df_input.mean())/ df_input.std()
    return df_input

plt.xticks([]) # 不顯示X軸
df_mean_norm = mean_norm(df["LOS"])

df_mean_norm.plot(kind='bar')

plt.show()