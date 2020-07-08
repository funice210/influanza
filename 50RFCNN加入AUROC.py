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


all_df = pd.read_csv('E:\\50RFCNN\\001RF_50_out.txt',sep = '\t',encoding = 'utf-8')
cols=['Label','ID',
'213988_s_at',
'201230_s_at',
'210592_s_at',
'202672_s_at',
'209906_at',
'202688_at',
'212206_s_at',
'212185_x_at',
'217964_at',
'212201_at',
'206513_at',
'204142_at',
'208751_at',
'218429_s_at',
'205660_at',
'209593_s_at',
'219684_at',
'218505_at',
'221050_s_at',
'35254_at',
'205698_s_at',
'208012_x_at',
'204972_at',
'208052_x_at',
'206461_x_at',
'202269_x_at',
'213851_at',
'217933_s_at',
'218280_x_at',
'205099_s_at',
'205241_at',
'203582_s_at',
'AFFX-HUMISGF3A/M97935_3_at',
'202380_s_at',
'202446_s_at',
'219627_at',
'221766_s_at',
'206133_at',
'211889_x_at',
'209093_s_at',
'209059_s_at',
'218559_s_at',
'203922_s_at',
'219716_at',
'204143_s_at',
'213294_at',
'208087_s_at',
'212681_at',
'206491_s_at',
'204711_at']

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
    
    
    Train4D_train_Features=train_Features.reshape(train_Features.shape[0],5,10,1).astype('float32')
    Test4D_test_Features=test_Features.reshape(test_Features.shape[0],5,10,1).astype('float32')
    
    train_LabelOneHot = np_utils.to_categorical(train_Label)
    test_LabelOneHot  = np_utils.to_categorical(test_Label)
    
    from keras.models import Sequential
    from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
    
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
    
    
    """all_Features,Label=PreprocessData(all_df)
    all_probability=model.predict(all_Features)
    pd=all_df
    pd.insert(len(all_df.columns),'probability',all_probability)"""
    
    #prediction=model.predict_classes((Test4D_test_Features)
    
      
    
    
    ##f1 = open('E:\Research\RFCNN\H3N2_RFCNN_prediction_100次.txt', 'a', encoding = 'UTF-8')
    #f2 = open('E:\Research\prediction100次結果\H3N2_CNN\H3N2_CNN_reality.txt_100次', 'a', encoding = 'UTF-8')    
    # 也可使用指定路徑等方式，如： C:\A.txt
    #W會覆寫 a才不會 
    
    """for p in predict:
        f1.write("%d\n" %(p))
    
    for o in test_Label:
        f2.write("%d\n" %(o))"""
    
 
    ##f1.write(str(TP)+" "+str(TN)+" "+str(FP)+" "+str(FN)+"\n")
    
    
    #f.write("%.3f\n" % (accuracy))
    ##f1.close
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
    
    #計算TPR,FPR
    def cal_rate(result, num, thres):  
        all_number = len(result[0])  
        # print all_number  
        TP = 0  
        FP = 0  
        FN = 0  
        TN = 0  
        for item in range(all_number):  
            disease = result[0][item,num]  
            if disease >= thres:  
                disease = 1  
            if disease == 1:  
                if result[1][item,num] == 1:  
                    TP += 1  
                else:  
                    FP += 1  
            else:  
                if result[1][item,num] == 0:  
                    TN += 1  
                else:  
                    FN += 1  
        # print TP+FP+TN+FN  
        accracy = float(TP+FP) / float(all_number)  
        if TP+FP == 0:  
            precision = 0  
        else:  
            precision = float(TP) / float(TP+FP)  
            TPR = float(TP) / float(TP+FN)  
            TNR = float(TN) / float(FP+TN)  
            FNR = float(FN) / float(TP+FN)  
            FPR = float(FP) / float(FP+TN)  
            # print accracy, precision, TPR, TNR, FNR, FPR  
            return accracy, precision, TPR, TNR, FNR, FPR
    #繪製AUROC    
    import numpy as np
    disease_class = ['H3N2']  
    style = ['r-']  
    ''''' 
    plot roc and calculate AUC/ERR, result: (prob, label)  
    '''  
    prob = np.random.rand(100,8)  
    label = np.where(prob>=0.5,prob,0)  
    label = np.where(label<0.5,label,1)  
    count = np.count_nonzero(label)  
    label = np.zeros((100,8))  
    label[1:20,:]=1  
    print (label) 
    print (prob)  
    print (count)
      
    for clss in range(len(disease_class)):  
        threshold_vaule = sorted(prob[:,clss])  
        threshold_num = len(threshold_vaule)  
        accracy_array = np.zeros(threshold_num)  
        precision_array = np.zeros(threshold_num)  
        TPR_array = np.zeros(threshold_num)  
        TNR_array = np.zeros(threshold_num)  
        FNR_array = np.zeros(threshold_num)  
        FPR_array = np.zeros(threshold_num)  
        # calculate all the rates  
        for thres in range(threshold_num):  
            accracy, precision, TPR, TNR, FNR, FPR = cal_rate((prob,label), clss, threshold_vaule[thres])  
            accracy_array[thres] = accracy  
            precision_array[thres] = precision  
            TPR_array[thres] = TPR  
            TNR_array[thres] = TNR  
            FNR_array[thres] = FNR  
            FPR_array[thres] = FPR  
        # print TPR_array  
        # print FPR_array  
        AUC = np.trapz(TPR_array, FPR_array)  
        threshold = np.argmin(abs(FNR_array - FPR_array))  
        EER = (FNR_array[threshold]+FPR_array[threshold])/2  
        print ('disease %10s threshold : %f' % (disease_class[clss],threshold))  
        print ('disease %10s accracy : %f' % (disease_class[clss],accracy_array[threshold]))  
        print ('disease %10s EER : %f AUC : %f' % (disease_class[clss],EER, -AUC))  
        plt.plot(FPR_array, TPR_array, style[clss], label=disease_class[clss])  
    plt.title('ROC')  
    plt.xlabel('False Positive Rate')  
    plt.ylabel('True Positive Rate')  
    plt.legend()  
    plt.show()