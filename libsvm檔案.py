# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:19:18 2018

@author: kc
"""

import numpy
import pandas as pd
from sklearn import cross_validation, svm, preprocessing, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve 
from sklearn.metrics import roc_auc_score

all_df = pd.read_csv('E:\\001_T0_rma.txt',sep = '\t',encoding = 'utf-8')

cols=['A/S','ID',
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
'202688_at']

all_df=all_df[cols]
#all_df = all_df[53:]
numpy.set_printoptions(threshold=numpy.inf, precision=3, suppress=True)


def PreprocessData(raw_df):
    df=raw_df.drop(['ID'], axis=1)#移除ID欄位
    ndarray = df.values#dataframe轉換為array
    Features = ndarray[:,1:]
    #Label = ndarray[:,0]
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures=minmax_scale.fit_transform(Features)
    numpy.reshape(scaledFeatures, (33,60))
    return scaledFeatures
    
all_df=PreprocessData(all_df)



f1 = open('E:\\H3N2minmax4225_2.txt', 'a', encoding = 'UTF-8')
#f2 = open('E:\Research\prediction100次結果\H3N2_CNN\H3N2_CNN_reality.txt_100次', 'a', encoding = 'UTF-8')    
# 也可使用指定路徑等方式，如： C:\A.txt
#W會覆寫 a才不會 
all_df=str(all_df.tolist())
f1.write(all_df)
    
    
    
#for o in test_Label:
    #f2.write("%d\n" %(o))
    
 
#f1.write(str(TP)+"\t"+str(TN)+"\t"+str(FP)+"\t"+str(FN)+"\t"+str(auc_roc)+"\t"+str(auc_pr)+"\n")
    
    
#f.write("%.3f\n" % (accuracy))
f1.close
#f2.close