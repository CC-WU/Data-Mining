# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:21:51 2021

@author: User
"""


import numpy as np 
import matplotlib.pyplot as plt # 畫出圖型 
import pandas as pd # 資料處理

# 顯示所有列
pd.set_option('display.max_columns', None)
# 顯示所有行
pd.set_option('display.max_rows', None)
# 不顯示X軸
plt.xticks([])
# 不顯示Y軸
plt.yticks([])

# Importing the dataset 
df = pd.read_excel("tr.xlsx", engine='openpyxl')
# df = pd.read_csv("final/tr.csv")
#print(df)

print("==================================================")

print(df.describe())
#df.boxplot()
#df.hist()

print("==================================================")

# 查看數據的大小
print(df.shape)

print("==================================================")

# 查看表中各變量的數據型態
print(df.dtypes)

print("==================================================")

# 判斷數據中是否存在重複值
print(df.duplicated().any())

print("==================================================")

# 去除重複值
print(df.drop_duplicates())

print("==================================================")


#==============================================================#

# distribution column
df_plot = df["Cancer history"] 
print(df_plot.value_counts()) # total 數量
print(df_plot.groupby(df_plot).value_counts()) # 各分類的數量

print(type(df_plot.groupby(df_plot).value_counts())) # 各分類的型態

print((df_plot.groupby(df_plot).value_counts()).idxmax()[0]) # 數量最多的類別名稱

print("==================================================")

#==============================================================#

# plot histogram 直方圖
if str(df_plot[0]).isdigit(): # numeric
    histogram = df_plot.plot.hist()
    # df['LOS'].plot(kind='bar')
    print(histogram)
    plt.show()
    
else: # category 
    df_plot.value_counts().plot(kind='bar')
    
print(df_plot.describe())

print("==================================================")
    
#==============================================================#

# 數據不平衡可視化

# df_majority = df[df.outcome==0]
# df_minority = df[df.outcome==1]

# df1 = df_majority
# df2 = df_minority
# plt.hist([df_majority.outcome, df_minority.outcome], label = ['0', '1'], stacked=True)
# plt.legend()
# plt.show()
