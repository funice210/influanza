# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 12:18:13 2018

@author: kc
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:32:12 2018

@author: yu-hao
目前:new50版本
read,cols要改
"""

# -*- coding: utf-8 -*-
import numpy
import pandas as pd
from sklearn import preprocessing
numpy.random.seed(10)

import rf_mlp_list
all_df = pd.read_csv("C:/Users/kc/Desktop/auto -rf/PROTO_prob_gene+state-rfs-f.txt",sep='\t',encoding='utf-8')
#all_df = pd.read_csv("C:/Users/yu-hao/Desktop/auto/48+2t.txt",sep='\t',encoding='utf-8')
cols=rf_mlp_list.cols



all_df=all_df[cols]
msk = numpy.random.rand(len(all_df)) < 0.8
#train_df = all_df[76:]
#test_df = all_df[:76]
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
test_Features ,test_Label =PreprocessData(test_df)
    
    
Train4D_train_Features=train_Features.reshape(train_Features.shape[0],5,10,1).astype('float32')
Test4D_test_Features=test_Features.reshape(test_Features.shape[0],5,10,1).astype('float32')
    
train_LabelOneHot = np_utils.to_categorical(train_Label)
test_LabelOneHot  = np_utils.to_categorical(test_Label)
    
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
model = Sequential()

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
import os

model = Sequential()
#建立卷積層1與池化層1
model.add(Conv2D(filters=16,#建立16個濾鏡filter weight
                 kernel_size=(3,3),#設定每個濾鏡3X3大小
                 padding='same',#此設定讓卷積運算，產生的卷積影像大小不變
                 input_shape=(5,10,1),#第1 2維度:代表輸入的影像形狀150x150大小，第3維度:最後維度是1
                 activation='relu'))#設定ReLU激活函數
model.add(MaxPooling2D(pool_size=(5,5)))#縮減取樣，將16個150x150影像，縮小為16個7x3影像
model.add(Dropout(rate=0.5))#訓練迭代時，隨機在神經網路中放棄50%的神經元

model.add(Conv2D(filters=32,#建立16個濾鏡filter weight
                 kernel_size=(15,15),#設定每個濾鏡15X15大小
                 padding='same',#此設定讓卷積運算，產生的卷積影像大小不變
                 input_shape=(1,2,1),#第1 2維度:代表輸入的影像形狀150x150大小，第3維度:最後維度是1
                 activation='relu'))#設定ReLU激活函數
model.add(MaxPooling2D(pool_size=(1,1)))#縮減取樣，將16個150x150影像，縮小為16個75x75影像
model.add(Dropout(0.5))#訓練迭代時，隨機在神經網路中放棄50%的神經元

    
    
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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])
train_history=model.fit(x=Train4D_train_Features,
                        y=train_LabelOneHot,validation_split=0.1, 
                         epochs=130,batch_size=50, verbose=0)
print(model.summary())
#test集準確率
score=model.evaluate(x=Test4D_test_Features,y=test_LabelOneHot)
print('test accuracy=',score[1])
    
    
    
all_Features,Label=PreprocessData(all_df)
all_Features=all_Features.reshape(all_Features.shape[0],5,10,1).astype('float32')
all_probability=model.predict_classes(all_Features)
predict=(all_probability[~msk])
print(predict)

import matplotlib.pyplot as plt

def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show()
	

#show_train_history(train_history,'acc','val_acc')
#show_train_history(train_history,'loss','val_loss')
print (scores[1])
#f = open('C:/Users/yu-hao/Desktop/H1N1/mrmr-result.txt', 'a', encoding = 'UTF-8')    # 也可使用指定路徑等方式，如： C:\A.txt
##W會覆寫 a才不會 
#a1=scores[1]
##f.write(str(a1)+"\n")
#f1 = open('C:/Users/yu-hao/Desktop/H1N1/mrmr_predicted.txt', 'a', encoding = 'UTF-8')
#f2 = open('C:/Users/yu-hao/Desktop/H1N1/mrmr_real.txt', 'a', encoding = 'UTF-8')    
## 也可使用指定路徑等方式，如： C:\A.txt
##W會覆寫 a才不會 
#
#for p in predict:
#    f1.write("%d\n" %(p))
#
#for o in test_Label:
#    f2.write("%d\n" %(o))
#
##f.write("%.3f\n" % (accuracy))
#f1.close
#f2.close    
#f1 = open('C:/Users/yu-hao/Desktop/H1N1/mrmr_tptn.txt', 'a', encoding = 'UTF-8')
f1 = open('C:/Users/kc/Desktop/auto -rf/rf_CNN_tptn50.txt', 'a', encoding = 'UTF-8')
   
# 也可使用指定路徑等方式，如： C:\A.txt
#W會覆寫 a才不會 

TP=0
TN=0
FP=0
FN=0
for j in range(predict.size):
    if(predict[j]==1 and predict[j]==test_Label[j]):
        TP=TP+1
    else:
        TP=TP+0
    
    if(predict[j]==0 and predict[j]==test_Label[j]):
        TN=TN+1
    else:
        TN=TN+0
    
    if(test_Label[j]==0 and predict[j]==1):
        FP=FP+1
    else:
        FP=FP+0
    
    if(test_Label[j]==1 and predict[j]==0):
        FN=FN+1
    else:
        FN=FN+0
print(TP,TN,FP,FN)
f1.write(str(TP)+"\t"+str(TN)+"\t"+str(FP)+"\t"+str(FN)+"\n")
f1.close