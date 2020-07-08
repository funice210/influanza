# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:28:17 2018

@author: kc
"""

import numpy
import pandas as pd
from keras.utils import np_utils
from sklearn import preprocessing
#numpy.random.seed(10)


all_df = pd.read_csv('E:\\probes_GSE52428matrix -100_out.txt',sep = '\t',encoding = 'utf-8')
cols=['Label','ID',
'202985_s_at',
'220784_s_at',
'212737_at',
'217860_at',
'216379_x_at',
'209422_at',
'207697_x_at',
'220051_at',
'205671_s_at',
'219382_at',
'219137_s_at',
'205905_s_at',
'213405_at',
'207815_at',
'35820_at',
'208866_at',
'209374_s_at',
'205821_at',
'218928_s_at',
'204142_at',
'217979_at',
'203467_at',
'201572_x_at',
'209828_s_at',
'219547_at',
'202630_at',
'210763_x_at',
'205708_s_at',
'203470_s_at',
'212527_at',
'204839_at',
'207759_s_at',
'210102_at',
'213348_at',
'217792_at',
'220068_at',
'204711_at',
'215838_at',
'218329_at',
'203620_s_at',
'215891_s_at',
'214022_s_at',
'207072_at',
'210789_x_at',
'209771_x_at',
'202178_at',
'213645_at',
'202306_at',
'204143_s_at',
'206478_at',
'203415_at',
'209994_s_at',
'217466_x_at',
'207819_s_at',
'210659_at',
'208594_x_at',
'212017_at',
'219532_at',
'217403_s_at',
'213567_at',
'211010_s_at',
'216733_s_at',
'212504_at',
'210968_s_at',
'202086_at',
'218276_s_at',
'48659_at',
'213215_at',
'203269_at',
'209920_at',
'205267_at',
'210993_s_at',
'202693_s_at',
'203113_s_at',
'222315_at',
'215211_at',
'219534_x_at',
'212945_s_at',
'221484_at',
'218458_at',
'209831_x_at',
'221688_s_at',
'204211_x_at',
'201230_s_at',
'200973_s_at',
'208914_at',
'211067_s_at',
'202981_x_at',
'208892_s_at',
'217968_at',
'219961_s_at',
'218421_at',
'206759_at',
'209117_at',
'209906_at',
'203521_s_at',
'213182_x_at',
'203246_s_at',
'210137_s_at',
'218170_at']

all_df=all_df[cols]
msk = numpy.random.rand(len(all_df)) < 0.8
train_df = all_df[msk]
test_df = all_df[~msk]
print('total:',len(all_df),
      'train:',len(train_df),
      'test:',len(test_df))

def PreprocessData(raw_df):
    df=raw_df.drop(['ID'], axis=1)#移除ID欄位
    ndarray = df.values#dataframe轉換為array
    Features = ndarray[:,1:] 
    Label = ndarray[:,0]

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures=minmax_scale.fit_transform(Features)    
    scaledFeatures[2:1]
    return scaledFeatures,Label
    
train_Features,train_Label=PreprocessData(train_df)
test_Features,test_Label=PreprocessData(test_df)

from keras.models import Sequential
from keras.layers import Dense,Dropout




model = Sequential()    
model.add(Dense(units=25, input_dim=100, 
                kernel_initializer='uniform', 
                activation='relu'))

model.add(Dense(units=10, 
                kernel_initializer='uniform', 
                activation='relu'))
model.add(Dense(units=1,
                kernel_initializer='uniform',
                activation='sigmoid'))
model.compile(loss='binary_crossentropy', 
              optimizer='adam', metrics=['accuracy'])
print(model.summary())
train_history =model.fit(x=train_Features, 
                         y=train_Label, 
                         validation_split=0.1, 
                         epochs=200, 
                         batch_size=400, verbose=2)





print(model.summary())
all_Features,Label=PreprocessData(all_df)
all_probability=model.predict(all_Features)
pd=all_df
pd.insert(len(all_df.columns),'probability',all_probability)

scores=model.evaluate(x=test_Features,y=test_Label)
print()
print('accuracy=',scores[1])

predict=(all_probability[~msk])
#print(predict)
    

import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel('accuacy')
    plt.xlabel('Epoch')
    plt.legend(['train','validation',],loc='upper left')
    plt.show()

show_train_history(train_history,'acc','val_acc')

def show_train_history1(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['train_loss','validation_loss',],loc='upper left')
    plt.show()

show_train_history1(train_history,'loss','val_loss')
del(model)