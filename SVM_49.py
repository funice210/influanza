# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:19:18 2018

@author: kc
"""

import numpy
import pandas as pd
from sklearn import cross_validation, svm, preprocessing, metrics

all_df = pd.read_csv('E:\\H3N2_samples_50\\001_50_out.txt',sep = '\t',encoding = 'utf-8')
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
'None probes_1']

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
# 建立訓練與測試資料


#titanic_X = pd.DataFrame(train_df).T
#titanic_y = train_df["Label"]
#train_X, test_X, train_y, test_y = cross_validation.train_test_split(titanic_X, titanic_y, test_size = 0.3)

# 建立 SVC 模型
svc = svm.SVC( probability = True ,C = 0.03125,  gamma=0.0078125)
svc_fit = svc.fit(train_Features, train_Label)

# 預測
test_y_predicted = svc.predict(test_Features)

# 績效
fpr, tpr, thresholds = metrics.roc_curve(test_Label, test_y_predicted)
auc = metrics.auc(fpr, tpr)
print(auc)
