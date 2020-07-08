# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 11:21:20 2018

@author: kc
"""

from keras.utils import np_utils
import numpy as np
np.random.seed(10)

import scipy.io as sio 
import matplotlib.pyplot as plt 
import numpy as np 
#matlab檔名 
matfn=u'C:/Users/kc/Desktop/deep learning_training data/test_data_ds1f.mat' 
test_data=sio.loadmat(matfn)
matfn=u'C:/Users/kc/Desktop/deep learning_training data/test_label_ds1f.mat' 
test_label=sio.loadmat(matfn)
matfn=u'C:/Users/kc/Desktop/deep learning_training data/train_data_ds1f.mat' 
train_data=sio.loadmat(matfn)
matfn=u'C:/Users/kc/Desktop/deep learning_training data/train_label_ds1f.mat' 
train_label=sio.loadmat(matfn)

print(test_label)
print(test_label.get('test_label'))

train_label=train_label.get('train_label')
test_label=test_label.get('test_label')
train_label=train_label[0][:]
test_label=test_label[0][:]

y_Train_OneHot = np_utils.to_categorical(train_label)
y_Test_OneHot = np_utils.to_categorical(test_label)

print(y_Test_OneHot)

x_Train =train_data.get('train_data').reshape(180, 1062000).astype('float32')
x_Test = test_data.get('test_data').reshape(20, 1062000).astype('float32')

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units=100, 
                input_dim=1062000, 
                kernel_initializer='normal', 
                activation='relu'))

model.add(Dense(units=2, 
                kernel_initializer='normal', 
                activation='softmax'))

print(model.summary())
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])

train_history=model.fit(x=x_Train,
                        y=y_Train_OneHot,validation_split=0.2, 
                        epochs=10, batch_size=10,verbose=2)