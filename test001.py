# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:46:28 2018

@author: kc
"""

import numpy
import pandas as pd
df1 = pd.read_excel("E:\\20180301-0531監測資料.xlsx","小1")
df1[:3]
cols=['水色','餵食量(kg)', '水溫', '鹽度','PH', 'NO2','排水']
df1=df1[cols]
df1[:3]

df1.isnull().sum()
df1['餵食量(kg)']=df1['餵食量(kg)'].fillna(0).astype(int)
df1['鹽度']=df1['鹽度'].fillna(33.7)
df1['NO2']=df1['NO2'].fillna(0.15)
df1['PH']=df1['PH'].fillna(7.72)
df1['水色']=df1['水色'].fillna('綠')
df1['水色']=df1['水色'].map({'綠':0,'褐':1}).astype(int)
df1['排水']=df1['排水'].map({'v':1})
df1['排水']=df1['排水'].fillna(0).astype(int)


ndarray=df1.values
ndarray.shape
ndarray[:3]

Label=ndarray[:,0]
Feature=ndarray[:,1:]
Label[:2]
Feature[:3]


def PreprocessData(raw_df):
    cols=['水色','餵食量(kg)', '水溫', '鹽度','PH', 'NO2','排水']
    df=raw_df[cols]
    
    df['餵食量(kg)']=df['餵食量(kg)'].fillna(0).astype(int)
    ph_mean = df['PH'].mean()
    df['PH'] = df['PH'].fillna(ph_mean)
    tmp_mean = df['水溫'].mean()
    df['水溫'] = df['水溫'].fillna(tmp_mean)
    sal_mean = df['鹽度'].mean()
    df['鹽度'] = df['鹽度'].fillna(sal_mean)
    
    df['NO2']=df['NO2'].fillna(0.15)
    df['水色']=df['水色'].fillna('綠')
    
    df['水色']=df['水色'].map({'綠':0,'褐':1}).astype(int)
    df['排水']=df['排水'].map({'v':1})
    df['排水']=df['排水'].fillna(0).astype(int)
    
    
from sklearn import preprocessing
minmax_scale=preprocessing.MinMaxScaler(feature_range=(0,1))
scaledFeatures=minmax_scale.fit_transform(Feature)
scaledFeatures[:3]
msk = numpy.random.rand(len(df1)) < 0.8
train_df = df1[msk]
test_df = df1[~msk]
print('total:',len(df1),
      'train:',len(train_df),
      'test:',len(test_df))



    ndarray = df.values
    Features = ndarray[:,1:]
    Label = ndarray[:,0]

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures=minmax_scale.fit_transform(Features)    
    
    return scaledFeatures,Label

train_Features,train_Label=PreprocessData(train_df)
test_Features,test_Label=PreprocessData(test_df)
train_Features[:2]
train_Label[:2]