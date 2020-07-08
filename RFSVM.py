# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:19:18 2018

@author: kc
"""

import numpy
import pandas as pd
from sklearn import cross_validation, svm, preprocessing, metrics
import rf_mlp_list
all_df = pd.read_csv("E:/auto -rf/PROTO_prob_gene+state-rfs-f.txt",sep='\t',encoding='utf-8')

cols=rf_mlp_list.cols


all_df=all_df[cols]

msk = numpy.random.rand(len(all_df)) < 0.8

train_df = all_df[msk]
test_df = all_df[~msk]

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


# 建立 SVC 模型 
svc = svm.SVC()
svc_fit = svc.fit(train_Features, train_Label)

# 預測
test_y_predicted = svc.predict(test_Features)

# 績效
accuracy = metrics.accuracy_score(test_Label, test_y_predicted)
print(accuracy)

fpr, tpr, thresholds = metrics.roc_curve(test_Label, test_y_predicted)
auc = metrics.auc(fpr, tpr)
print(auc)
TP=0
TN=0
FP=0
FN=0
for j in range(test_y_predicted.size):
    if(test_y_predicted[j]==1 and test_y_predicted[j]==test_Label[j]):
        TP=TP+1
    else:
        TP=TP+0
        
    if(test_y_predicted[j]==0 and test_y_predicted[j]==test_Label[j]):
        TN=TN+1
    else:
        TN=TN+0
        
    if(test_Label[j]==0 and test_y_predicted[j]==1):
        FP=FP+1
    else:
        FP=FP+0
        
    if(test_Label[j]==1 and test_y_predicted[j]==0):
        FN=FN+1
    else:
        FN=FN+0
print(TP,TN,FP,FN)
    
f1 = open('E:\Research\prediction100次結果\SVM\H3N2_RFSVM50_all_prediction_P0_100次.txt', 'a', encoding = 'UTF-8')
#f2 = open('E:\Research\prediction100次結果\H3N2_CNN\H3N2_CNN_reality.txt_100次', 'a', encoding = 'UTF-8')    
# 也可使用指定路徑等方式，如： C:\A.txt
#W會覆寫 a才不會 
    
"""for p in predict:
    f1.write("%d\n" %(p))
    
for o in test_Label:
    f2.write("%d\n" %(o))"""
    
 
f1.write(str(TP)+"\t"+str(TN)+"\t"+str(FP)+"\t"+str(FN)+"\n")
    
    
#f.write("%.3f\n" % (accuracy))
f1.close
#f2.close
   
    