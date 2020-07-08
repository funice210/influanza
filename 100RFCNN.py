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


all_df = pd.read_csv('E:\\100RFCNN\\001RF_100_out.txt',sep = '\t',encoding = 'utf-8')
cols=['Label','ID',
'213293_s_at',
'204211_x_at',
'219684_at',
'205552_s_at',
'221044_s_at',
'34689_at',
'204711_at',
'209906_at',
'208912_s_at',
'211794_at',
'213988_s_at',
'213851_at',
'209969_s_at',
'207500_at',
'209093_s_at',
'202269_x_at',
'210592_s_at',
'209994_s_at',
'203595_s_at',
'205170_at',
'208436_s_at',
'202446_s_at',
'208751_at',
'65472_at',
'201230_s_at',
'208087_s_at',
'207697_x_at',
'205821_at',
'202145_at',
'211666_x_at',
'200986_at',
'203127_s_at',
'212658_at',
'201272_at',
'202656_s_at',
'205483_s_at',
'207000_s_at',
'207574_s_at',
'200610_s_at',
'211012_s_at',
'204439_at',
'40569_at',
'210817_s_at',
'204747_at',
'218400_at',
'208631_s_at',
'209417_s_at',
'206461_x_at',
'209828_s_at',
'204804_at',
'210202_s_at',
'222034_at',
'202087_s_at',
'218543_s_at',
'205099_s_at',
'220146_at',
'214163_at',
'217964_at',
'217378_x_at',
'208581_x_at',
'209640_at',
'200034_s_at',
'204994_at',
'209305_s_at',
'202414_at',
'220252_x_at',
'202649_x_at',
'210845_s_at',
'219863_at',
'202307_s_at',
'205266_at',
'210763_x_at',
'204352_at',
'202864_s_at',
'218149_s_at',
'200768_s_at',
'204199_at',
'204423_at',
'205055_at',
'202086_at',
'202464_s_at',
'208892_s_at',
'200858_s_at',
'210223_s_at',
'215460_x_at',
'219014_at',
'221865_at',
'210430_x_at',
'205660_at',
'211941_s_at',
'201600_at',
'202358_s_at',
'212380_at',
'218809_at',
'35820_at',
'206777_s_at',
'205698_s_at',
'56256_at',
'202869_at',
'218774_at']

all_df=all_df[cols]

i=0
while i<1:   
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
    
    
    Train4D_train_Features=train_Features.reshape(train_Features.shape[0],10,10,1).astype('float32')
    Test4D_test_Features=test_Features.reshape(test_Features.shape[0],10,10,1).astype('float32')
    
    train_LabelOneHot = np_utils.to_categorical(train_Label)
    test_LabelOneHot  = np_utils.to_categorical(test_Label)
    
    from keras.models import Sequential
    from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
    
    model = Sequential()
    #建立卷積層1與池化層1
    model.add(Conv2D(filters=16,#建立16個濾鏡filter weight
                     kernel_size=(3,3),#設定每個濾鏡3X3大小
                     padding='same',#此設定讓卷積運算，產生的卷積影像大小不變
                     input_shape=(10,10,1),#第1 2維度:代表輸入的影像形狀150x150大小，第3維度:最後維度是1
                     activation='relu'))#設定ReLU激活函數
    model.add(MaxPooling2D(pool_size=(2,2)))#縮減取樣，將16個150x150影像，縮小為16個7x3影像
    model.add(Dropout(rate=0.5))#訓練迭代時，隨機在神經網路中放棄50%的神經元
    
    model.add(Conv2D(filters=32,#建立16個濾鏡filter weight
                     kernel_size=(15,15),#設定每個濾鏡15X15大小
                     padding='same',#此設定讓卷積運算，產生的卷積影像大小不變
                     input_shape=(5,5,1),#第1 2維度:代表輸入的影像形狀150x150大小，第3維度:最後維度是1
                     activation='relu'))#設定ReLU激活函數
    model.add(MaxPooling2D(pool_size=(5,5)))#縮減取樣，將16個150x150影像，縮小為16個75x75影像
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


    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])
    train_history=model.fit(x=Train4D_train_Features,
                            y=train_LabelOneHot,validation_split=0.1, 
                             epochs=130,batch_size=50, verbose=2)
    print(model.summary())
    #test集準確率
    score=model.evaluate(x=Test4D_test_Features,y=test_LabelOneHot)
    print('test accuracy=',score[1])
    
    
    
    all_Features,Label=PreprocessData(all_df)
    all_Features=all_Features.reshape(all_Features.shape[0],10,10,1).astype('float32')
    all_probability=model.predict_classes(all_Features)
    predict=(all_probability[~msk])
    print(predict)
    
    
    """all_Features,Label=PreprocessData(all_df)
    all_probability=model.predict(all_Features)
    pd=all_df
    pd.insert(len(all_df.columns),'probability',all_probability)"""
    
    #prediction=model.predict_classes((Test4D_test_Features)
    
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
    
    #f1 = open('E:\Research\RFCNN\H3N2_RFCNN100_prediction_100次.txt', 'a', encoding = 'UTF-8')
    #f2 = open('E:\Research\prediction100次結果\H3N2_CNN\H3N2_CNN_reality.txt_100次', 'a', encoding = 'UTF-8')    
    # 也可使用指定路徑等方式，如： C:\A.txt
    #W會覆寫 a才不會 
    
    """for p in predict:
        f1.write("%d\n" %(p))
    
    for o in test_Label:
        f2.write("%d\n" %(o))"""
    
 
    #f1.write(str(TP)+" "+str(TN)+" "+str(FP)+" "+str(FN)+"\n")
    
    
    #f.write("%.3f\n" % (accuracy))
    #f1.close
    #f2.close
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
    i+=1
    print(i)