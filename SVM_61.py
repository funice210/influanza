# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:09:54 2018

@author: kc
"""

import numpy
import pandas as pd
from sklearn import cross_validation, svm, preprocessing, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve 
from sklearn.metrics import roc_auc_score
#numpy.random.seed(10)


all_df = pd.read_csv('E:\\H3N2_samples_61\\001_61_P0_out.txt',sep = '\t',encoding = 'utf-8')
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
'221728_x_at',
'None probes_1',
'None probes_2']

all_df=all_df[cols]
i=0
while i<101:   
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
    svc = svm.SVC( probability = True ,C = 8192.0,  gamma=0.003125)
    svc_fit = svc.fit(train_Features, train_Label)
    
    # 預測
    test_y_predicted = svc.predict(test_Features)
    
    #roc
    fpr, tpr, thresholds = metrics.roc_curve(test_Label, test_y_predicted)
    auc_roc = metrics.auc(fpr, tpr)
    #pr
    
    precision, recall, thresholds = precision_recall_curve(test_Label, test_y_predicted)
    auc_pr = metrics.auc(recall, precision)
    
    print("auc(pr)",auc_pr)
    print("auc(roc)",auc_roc)
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
        
    f1 = open('E:\Research\prediction100次結果\SVM\H3N2_SVM_61_prediction_P0_100次_0311.txt', 'a', encoding = 'UTF-8')
    #f2 = open('E:\Research\prediction100次結果\H3N2_CNN\H3N2_CNN_reality.txt_100次', 'a', encoding = 'UTF-8')    
    # 也可使用指定路徑等方式，如： C:\A.txt
    #W會覆寫 a才不會 
        
    """for p in predict:
        f1.write("%d\n" %(p))
        
    for o in test_Label:
        f2.write("%d\n" %(o))"""
        
     
    f1.write(str(TP)+"\t"+str(TN)+"\t"+str(FP)+"\t"+str(FN)+"\t"+str(auc_roc)+"\t"+str(auc_pr)+"\n")
        
        
    #f.write("%.3f\n" % (accuracy))
    f1.close
    #f2.close
    i+=1
    print(i)
        
        