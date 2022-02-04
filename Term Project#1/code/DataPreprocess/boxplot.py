# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 00:45:53 2021

@author: User
"""

# 導入第三方模塊 
import pandas as pd 
import matplotlib.pyplot as plt 

# 導入數據 
sunspots = pd.read_excel("tr.xlsx", engine='openpyxl')
print(sunspots)

# 繪製箱線圖（1.5倍的四分位差，如需繪製3倍的四分位差，只需調整whis參數） 
sunspots.boxplot() 
plt.xticks(rotation=90)

# 顯示圖形 
plt.show()

# 計算下四分位數和上四分位 
Q1 = sunspots["GOT"].quantile(q = 0.25) 
Q3 = sunspots["GOT"].quantile(q = 0.75) 
print(Q1)
print(Q3)

# 基於1.5倍的四分位差計算上下須對應的值 
low_whisker = Q1 - 1.5*(Q3 - Q1) 
up_whisker = Q3 + 1.5*(Q3 - Q1) 
print(low_whisker)
print(up_whisker)

# 尋找異常點 
print(sunspots.value_counts()[(sunspots.value_counts() > up_whisker) | (sunspots.value_counts() < low_whisker)])



