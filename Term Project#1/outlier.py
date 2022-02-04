# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 03:02:44 2021

@author: User
"""

# 導入第三方模塊 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
plt.rc("font",family="SimHei",size="15") # 解決中文亂碼問題
import numpy as np

# 導入數據 
sunspots = pd.read_excel("tr.xlsx", engine='openpyxl')
print(sunspots)


tipmean=sunspots['GOT'].mean()
tipstd = sunspots['GOT'].std()
topnum1 = tipmean + 2 * tipstd
bottomnum1 = tipmean - 2 * tipstd
print(sunspots.head(10))
print("正常值的範圍：",topnum1,bottomnum1)
print("是否存在超出正常範圍的值：",any(sunspots['GOT']>topnum1))
print("是否存在小于正常範圍的值：",any(sunspots['GOT']<bottomnum1))



mean1 = sunspots['GOT'].quantile(q=0.25)#下四分位差
mean2 = sunspots['GOT'].quantile(q=0.75)#上四分位差
mean3 = mean2-mean1#中位差
topnum2 = mean2 + 1.5 * mean3
bottomnum2 = mean2 - 1.5 * mean3
print("正常值的範圍：",topnum2,bottomnum2)
print("是否存在超出正常範圍的值：",any(sunspots['GOT']>topnum2))
print("是否存在小于正常範圍的值：",any(sunspots['GOT']<bottomnum2))

fig, ax =plt.subplots(1,2)
sns.boxplot(x=sunspots["GOT"],data=sunspots, ax=ax[0])

#plt.boxplot(x=sunspots['GOT'])
#plt.show()

# 修改離異值
replace_value1 = sunspots['GOT'][sunspots['GOT'] < topnum2].max()
sunspots.loc[sunspots['GOT'] > topnum2,'GOT'] = replace_value1
replace_value2 = sunspots['GOT'][sunspots['GOT'] > bottomnum2].min()
sunspots.loc[sunspots['GOT'] < bottomnum2,'GOT'] = replace_value2


sns.boxplot(x=sunspots["GOT"], data=sunspots, ax=ax[1])

#fig1, ax1 = plt.subplots()
#ax1.set_title('Basic Plot')
#ax1.boxplot(sunspots["GOT"])