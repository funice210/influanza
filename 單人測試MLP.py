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
from sklearn import cross_validation, ensemble, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
numpy.random.seed(10)

import rf_mlp_list_Teacher
all_df = pd.read_csv("E:\\auto -rf\\001\\001_T0_rma.txt",sep='\t',encoding='utf-8')
#all_df = pd.read_csv("C:/Users/yu-hao/Desktop/auto/48+2t.txt",sep='\t',encoding='utf-8')
cols=rf_mlp_list_Teacher.cols



all_df=all_df[cols]
#msk = numpy.random.rand(len(all_df)) < 0.8
train_df = all_df[2:]
test_df = all_df[:2]
#train_df = all_df[msk]
#test_df = all_df[~msk]
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
    
    return scaledFeatures,Label

#train_Features,train_Label=PreprocessData(train_df)
#test_Features,test_Label=PreprocessData(test_df)

#minmax 全資料
all_Features,all_Label=PreprocessData(all_df)
train_Features=all_Features[2:]
train_Label=all_Label[2:]
test_Features=all_Features[:2]
test_Label=all_Label[:2]



from keras.models import Sequential
from keras.layers import Dense,Dropout
model = Sequential()
#model.add(Dense(units=50, input_dim=50, 
#                kernel_initializer='uniform', 
#                activation='relu'))
#
#model.add(Dense(units=30,
#                kernel_initializer='uniform', 
#                activation='relu'))
#model.add(Dense(units=30,
#                kernel_initializer='uniform', 
#                activation='relu'))
model.add(Dense(units=100, input_dim=295, 
                kernel_initializer='uniform', 
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=50,
                kernel_initializer='uniform', 
                activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1, 
                kernel_initializer='uniform',
                activation='sigmoid'))
model.summary()


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config)) 


model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
train_history =model.fit(x=train_Features, 
                         y=train_Label, 
                         validation_split=0.1, 
                         epochs=200, 
                         batch_size=100,verbose=0)


#test集準確率
#score=model.evaluate(x=Test4D_test_Features,y=test_LabelOneHot)
#print('test accuracy=',score[1])
    
    
    
all_Features,Label=PreprocessData(all_df)
all_probability=model.predict_classes(all_Features)
pd=all_df
pd.insert(len(all_df.columns),
          'probability',all_probability)
predict=all_probability[:2]
predict_probability=model.predict_proba(all_Features)
predict_probability=predict_probability[:2]

#predict=all_probability[~msk]
scores=model.evaluate(x=test_Features,
                      y=test_Label)






"""import matplotlib.pyplot as plt
def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show()"""
	

#show_train_history(train_history,'acc','val_acc')
#show_train_history(train_history,'loss','val_loss')
#print (scores[1])
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
"""f1 = open('E:\\RMA實驗\\RFMLP\\T0\\teacher_select_new_001_預測T0.txt', 'a', encoding = 'UTF-8')
f2 = open('E:\\RMA實驗\\RFMLP\\T0\\teacher_select_new_001_單人樣本T0.txt', 'a', encoding = 'UTF-8')
   
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
#f1.write(str(TP)+"\t"+str(TN)+"\t"+str(FP)+"\t"+str(FN)+"\n")
f1.write(str(TP)+"\t"+str(TN)+"\t"+str(FP)+"\t"+str(FN)+"\n")
predict=predict.tolist()
predict_probability=predict_probability.tolist()
f2.write(str(predict)+"\t"+str(predict_probability)+"\n")
f1.close
f2.close"""