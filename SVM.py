# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:19:18 2018

@author: kc
"""

import numpy as np
import pandas as pd
from sklearn import cross_validation, svm, preprocessing, metrics

# 載入資料
url = "https://storage.googleapis.com/2017_ithome_ironman/data/kaggle_titanic_train.csv"
titanic_train = pd.read_csv(url)

# 填補遺漏值
age_median = np.nanmedian(titanic_train["Age"])
new_Age = np.where(titanic_train["Age"].isnull(), age_median, titanic_train["Age"])
titanic_train["Age"] = new_Age

# 創造 dummy variables
label_encoder = preprocessing.LabelEncoder()
encoded_Sex = label_encoder.fit_transform(titanic_train["Sex"])

# 建立訓練與測試資料
titanic_X = pd.DataFrame([titanic_train["Pclass"],
                         encoded_Sex,
                         titanic_train["Age"]
]).T
titanic_y = titanic_train["Survived"]
train_X, test_X, train_y, test_y = cross_validation.train_test_split(titanic_X, titanic_y, test_size = 0.3)

# 建立 SVC 模型
svc = svm.SVC()
svc_fit = svc.fit(train_X, train_y)

# 預測
test_y_predicted = svc.predict(test_X)

# 績效
fpr, tpr, thresholds = metrics.roc_curve(test_y, test_y_predicted)
auc = metrics.auc(fpr, tpr)
print(auc)
