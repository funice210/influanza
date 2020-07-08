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


all_df = pd.read_csv('E:\\H3N2_samples_61\\001_out61.txt',sep = '\t',encoding = 'utf-8')
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
'222154_s_at',
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
'214453_s_at',
'207353_s_at',
'210797_s_at',
'44673_at',
'203596_s_at',
'202270_at',
'219209_at',
'213294_at',
'206026_s_at',
'202688_at',
'221728_x_at']

all_df=all_df[cols]

def PreprocessData(raw_df):
   
    df=raw_df.drop(['ID'], axis=1)#移除name欄位
    ndarray = df.values#dataframe轉換為array
    Features = ndarray[:,1:] 
    Label = ndarray[:,0]

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures=minmax_scale.fit_transform(Features)    
    
    return scaledFeatures,Label


                #for i in range(80):
                
            
all_df=all_df[cols]
msk = numpy.random.rand(len(all_df)) < 0.8
#前16筆資料為測試集，16之後為訓練集
#train_df = all_df[16:]
#test_df = all_df[:16]
train_df = all_df[msk]
test_df = all_df[~msk]
print('total:',len(all_df),
       'train:',len(train_df),
       'test:',len(test_df))
#定義feature及label
train_Features,train_Label=PreprocessData(train_df)
test_Features,test_Label=PreprocessData(test_df)
   
from keras.models import Sequential
from keras.layers import Dense,Dropout
    
#建立模型
model = Sequential()
#輸入層
model.add(Dense(units=49, input_dim=63, 
                kernel_initializer='uniform', 
                activation='relu'))
model.add(Dropout(0.1))
#隱藏層
model.add(Dense(units=36,
                kernel_initializer='uniform', 
                activation='relu'))
  
    
    
model.add(Dropout(0.5))
#輸出層
   
model.add(Dense(units=1, 
                kernel_initializer='uniform',
                activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
#輸入參數
train_history =model.fit(x=train_Features, 
                         y=train_Label, 
                         validation_split=0.1, 
                         epochs=100, 
                         batch_size=50,verbose=0)
 
  
    
all_Features,Label=PreprocessData(all_df)
all_probability=model.predict_classes(all_Features)
pd=all_df
pd.insert(len(all_df.columns),
          'probability',all_probability)
predict=all_probability[~msk]
#print(all_probability[~msk])
  
    
    
    
#計算測試準確率
scores=model.evaluate(x=test_Features,y=test_Label)
#畫圖
import matplotlib.pyplot as plt
def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show()
  
    
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')
print (scores[1])
#f = open('C:/Users/yu-hao/Desktop/H1N1/result.txt', 'a', encoding = 'UTF-8')    # 也可使用指定路徑等方式，如： C:\A.txt
    ##W會覆寫 a才不會 
    #a1=scores[1]
    #f.write(str(a1)+"\n")
    #f1 = open('C:/Users/yu-hao/Desktop/H1N1/all_predicted.txt', 'a', encoding = 'UTF-8')
    #f2 = open('C:/Users/yu-hao/Desktop/H1N1/all_real.txt', 'a', encoding = 'UTF-8')    
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
    #f1 = open('C:/Users/yu-hao/Desktop/H1N1/all_tptn.txt', 'a', encoding = 'UTF-8')
f1 = open('E:\H3N2MLP\H3N2_tptn61.txt', 'a', encoding = 'UTF-8')
f2 = open('E:\H3N2MLP\H3N2_tptn61_predict.txt', 'a', encoding = 'UTF-8') 
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
f2.write(str(predict)+"\t"+str(test_Label)+"\n")
  
#f.write("%.3f\n" % (accuracy))
f1.close
f2.close
i+=1 
print(i)   