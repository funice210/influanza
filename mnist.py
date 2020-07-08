# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 17:22:49 2018

@author: kc
"""

from keras.utils import np_utils
import numpy as np
np.random.seed(10)

from keras.datasets import mnist
(x_train_image, y_train_label), \
(x_test_image, y_test_label) = mnist.load_data()

x_Train =x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')

x_Train_normalize = x_Train/ 255
x_Test_normalize = x_Test/ 255

y_Train_OneHot = np_utils.to_categorical(y_train_label)
y_Test_OneHot = np_utils.to_categorical(y_test_label)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential()
#將"輸入層"與"隱藏層1"加入模型
model.add(Dense(units=500,input_dim=784,kernel_initializer='normal',activation='relu'))
#DropOut避免Overfitting
model.add(Dropout(0.5))
#將"隱藏層2"加入模型
model.add(Dense(units=500,kernel_initializer='normal',activation='relu'))
#DropOut避免Overfitting
model.add(Dropout(0.5))
#將"輸出層"加入模型
model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

train_history=model.fit(x=x_Train_normalize,
                        y=y_Train_OneHot,validation_split=0.2,
                        epochs=200,batch_size=100,verbose=2)

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

#show_train_history(train_history,'loss','val_loss')
scores = model.evaluate(x_Test_normalize,y_Test_OneHot)
print()
print('accuracy=',scores[1])

prediction=model.predict_classes(x_Test)

def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1+i)
        ax.imshow(images[idx], cmap='binary')
        title = "label=" +str(labels[idx])
        if len (prediction)>0:
            title+=",predict="+str(prediction[idx])
        ax.set_title(title,fontsize = 10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()    
#plot_images_labels_prediction(x_test_image,y_test_label,prediction,idx=340)
#查詢資料
#import pandas as pd
#pd.crosstab(y_test_label,prediction,colnames=['predict'],rownames=['label'])
#df=pd.DataFrame({'label':y_test_label, 'predict':prediction})
#df=[(df.label==5)&(df.predict==3)]

