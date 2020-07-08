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


all_df = pd.read_csv('E:\\H3N2_samples_50\\001_50_out.txt',sep = '\t',encoding = 'utf-8')
cols=['Label','ID',
'213797_at',
'203153_at',
'204439_at',
'219863_at',
'204439_at',
'218400_at',
'202086_at',
'204747_at',
'205483_s_at',
'200986_at',
'217502_at',
'205660_at',
'205569_at',
'204415_at',
'205552_s_at',
'206553_at',
'202411_at',
'206133_at',
'215856_at',
'218943_s_at',
'203595_s_at',
'218986_s_at',
'212203_x_at',
'202145_at',
'202269_x_at',
'216020_at',
'204211_x_at',
'217933_s_at',
'219062_s_at',
'209417_s_at',
'208436_s_at',
'202430_s_at',
'206025_s_at',
'218543_s_at',
'219684_at',
'219352_at',
'AFFX-HUMISGF3A/M97935_3_at',
'205241_at',
'208087_s_at',
'201649_at',
'202687_s_at',
'213293_s_at',
'221950_at',
'209593_s_at',
'219716_at',
'220358_at',
'204533_at',
'204606_at',
'214218_s_at',
'None probes_1']

all_df=all_df[cols]

msk = numpy.random.rand(len(all_df)) < 0.8

train_df = all_df[msk]
test_df = all_df[~msk]

#print(test_df)
#train_df = all_df[16:]
#test_df = all_df[:16]

#print(test_df)
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
test_Features ,test_Label =PreprocessData(test_df)
#print(train_Label)

Train4D_train_Features=train_Features.reshape(train_Features.shape[0],5,10,1).astype('float32')
Test4D_test_Features=test_Features.reshape(test_Features.shape[0],5,10,1).astype('float32')
#print(Train4D_train_Features.shape)
#print(Test4D_test_Features.shape)
train_LabelOneHot = np_utils.to_categorical(train_Label)
test_LabelOneHot  = np_utils.to_categorical(test_Label)

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
#import tensorflow as tf
#import keras.backend.tensorflow_backend as KTF

#KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'cpu':0})))
model = Sequential()
#建立卷積層1與池化層1


model.add(Conv2D(filters=16,#建立16個濾鏡filter weight
                 kernel_size=(3,3),#設定每個濾鏡3X3大小
                 padding='same',#此設定讓卷積運算，產生的卷積影像大小不變
                 input_shape=(5,10,1),#第1 2維度:代表輸入的影像形狀150x150大小，第3維度:最後維度是1
                 activation='relu'))#設定ReLU激活函數
model.add(MaxPooling2D(pool_size=(5,5)))#縮減取樣，將16個150x150影像，縮小為16個75x75影像
model.add(Dropout(rate=0.5))#訓練迭代時，隨機在神經網路中放棄50%的神經元

model.add(Conv2D(filters=32, 
                 kernel_size=(1, 1), 
                 activation='relu', 
                 padding='same'))
model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Dropout(0.5))


#建立平坦層
model.add(Flatten())#將16個75X75影像，轉換為1維向量，長度是16X75X75=90000
#建立隱藏層
model.add(Dense(100, kernel_initializer='uniform', activation='relu'))#共有100個神經元
model.add(Dropout(0.25))#訓練迭代時，隨機在神經網路中放棄25%的神經元，避免overfitting
model.add(Dense(100, kernel_initializer='uniform', activation='relu'))#共有100個神經元
model.add(Dropout(0.25))#訓練迭代時，隨機在神經網路中放棄25%的神經元，避免overfitting
model.add(Dense(100, kernel_initializer='uniform', activation='relu'))#共有100個神經元
#建立輸出層
model.add(Dense(2, kernel_initializer='uniform', activation='softmax'))#輸出2個神經元，使用softmax激活函數
print(model.summary())


model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])
train_history=model.fit(x=Train4D_train_Features,
                        y=train_LabelOneHot,validation_split=0.1, 
                         epochs=100,batch_size=10, verbose=0)

#test集準確率
score=model.evaluate(x=Test4D_test_Features,y=test_LabelOneHot)
print('test accuracy=',score[1])



all_Features,Label=PreprocessData(all_df)
all_Features=all_Features.reshape(all_Features.shape[0],5,10,1).astype('float32')
all_probability=model.predict_classes(all_Features)
predict=(all_probability[~msk])
print(predict)


"""all_Features,Label=PreprocessData(all_df)
all_probability=model.predict(all_Features)
pd=all_df
pd.insert(len(all_df.columns),'probability',all_probability)"""

#prediction=model.predict_classes((Test4D_test_Features)


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
#del(model)
