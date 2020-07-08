# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:59:39 2018

@author: kc
"""

import pandas as pd
df = pd.read_excel('E:/metrix.xlsx',0, header = False) #读取文件 比如 df = pd.read_excel('C:/your_data.xlsx',0, header = False)
df_T = df.T #获得矩阵的转置
df_T.to_excel('E:/test.csv', sheet_name='sheet 1') #保存文件 比如 df_T.to_excel('C:/test.xlsx', sheet_name='sheet 1')

