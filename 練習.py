# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:54:09 2018

@author: kc
"""

from keras.datasets import cifar10
import numpy as np
np.random.seed(10)

(x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()

print("train data:",'images:',x_img_train.shape,"labels:",y_label_train.shape)
print("test data:",'images:',x_img_test.shape,"labels:",y_label_test.shape)

x_img_train_normalize=x_img_train.astype('float32')/255.0
x_img_test_normalize=x_img_test.astype('float32')/255.0

from keras.utils import np_utils
y_label_train_OneHot=np_utils.to_categorical(y_label_train)
y_label_test_OneHot=np_utils.to_categorical(y_label_test)

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D

import keras
print('keras: %s' % keras.__version__)

model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),
                 input_shape=(32,32,3),
                 activation="relu",
                 padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(3,3),
                       activation="relu",padding='same'))
model.add(Conv2D(filters=64,kernel_size=(3,3),
                 activation="relu",padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(rate=0.25))
model.add(Dense(1024,activation="relu"))
model.add(Dropout(rate=0.25))
model.add(Dense(10,activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimzer='adam',metrics=['accuracy'])
train_history=model.fit(x_img_train_normalize,y_label_train_OneHot,
                        validation_spilt=0.2,
                        epochs=10,batch_size=128,verbose=2)
show_train_history('acc','val_acc')
show_train_history('loss','val_loss')
score=model.evalute(x_img_test_normalize,
                    y_label_test_OneHot,verbose=0)