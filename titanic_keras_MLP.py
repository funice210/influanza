# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:30:59 2018

@author: kc
"""

import numpy
import pandas as pd
from sklearn import preprocessing
numpy.random.seed(10)


all_df = pd.read_excel("E:/titanic.xls")
cols=['survived','name','pclass' ,'sex', 'age', 'sibsp',
      'parch', 'fare', 'embarked']
all_df=all_df[cols]
msk = numpy.random.rand(len(all_df)) < 0.8
train_df = all_df[msk]
test_df = all_df[~msk]
print('total:',len(all_df),
      'train:',len(train_df),
      'test:',len(test_df))

def PreprocessData(raw_df):
    df=raw_df.drop(['name'], axis=1)#移除name欄位
    age_mean = df['age'].mean()#計算age欄位的平均值
    df['age'] = df['age'].fillna(age_mean)#將null值填入平均值
    fare_mean = df['fare'].mean()#計算fare欄位的平均值
    df['fare'] = df['fare'].fillna(fare_mean)#將null值填入平均值
    df['sex']= df['sex'].map({'female':0, 'male': 1}).astype(int)#文字轉為0與1
    x_OneHot_df = pd.get_dummies(data=df,columns=["embarked" ])#Embarked以Onehot Encoding轉換

    ndarray = x_OneHot_df.values#dataframe轉換為array
    Features = ndarray[:,1:] 
    Label = ndarray[:,0]

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures=minmax_scale.fit_transform(Features)    
    
    return scaledFeatures,Label

train_Features,train_Label=PreprocessData(train_df)
test_Features,test_Label=PreprocessData(test_df)

from keras.models import Sequential
from keras.layers import Dense,Dropout
model = Sequential()
model.add(Dense(units=40, input_dim=9, 
                kernel_initializer='uniform', 
                activation='relu'))
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
                         epochs=50, 
                         batch_size=30,verbose=2)

Jack=pd.Series([0,'Jack',3,'male',23,1,0,5.0000,'S'])
Rose=pd.Series([1,'Rose',1,'female',20,1,0,100.0000,'S']) 
JR_df=pd.DataFrame([list(Jack),list(Rose)],
                    columns=['survived','name','pclass','sex',
                             'age','sibsp','parch','fare','embarked'])
all_df=pd.concat([all_df,JR_df])

print(all_df[-2:])


all_Features,Label=PreprocessData(all_df)
all_probability=model.predict(all_Features)
print(all_probability[:10])


pd=all_df
pd.insert(len(all_df.columns),
          'probability',all_probability)
print(pd[-2:])


scores=model.evaluate(x=test_Features,
                     y=test_Label)

import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show

show_train_history(train_history,'acc','val_acc')
scores[1]

