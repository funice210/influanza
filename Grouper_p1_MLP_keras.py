# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 15:12:53 2018

@author: kc
"""

import numpy
import pandas as pd
from sklearn import preprocessing
numpy.random.seed(10)
df1 = pd.read_excel("E:\\all2.xlsx")
df1.isnull().sum()

msk = numpy.random.rand(len(df1)) < 0.8
train_df = df1[msk]
test_df = df1[~msk]

print('total:',len(df1),
      'train:',len(train_df),
      'test:',len(test_df))

def PreprocessData(df):

    ndarray = df.values
    Features = ndarray[:,1:]
    Label = ndarray[:,0]

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures=minmax_scale.fit_transform(Features)    
    
    return scaledFeatures,Label

train_Features,train_Label=PreprocessData(train_df)
test_Features,test_Label=PreprocessData(test_df)
print(train_Features)
from keras.models import Sequential
from keras.layers import Dense,Dropout
model = Sequential()
model.add(Dense(units=20, 
                input_dim=7,
                kernel_initializer='uniform',
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=30, 
                kernel_initializer='uniform', 
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=30, 
                kernel_initializer='uniform', 
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=30, 
                kernel_initializer='uniform', 
                activation='relu'))
model.add(Dense(units=1, 
                kernel_initializer='uniform',
                activation='sigmoid'))
model.compile(loss='binary_crossentropy', 
              optimizer='adam', metrics=['accuracy'])

train_history =model.fit(x=train_Features, 
                         y=train_Label, 
                         validation_split=0.1, 
                         epochs=100, 
                         batch_size=50,verbose=2)

import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')

scores = model.evaluate(x=test_Features, 
                        y=test_Label)

print(scores[1])