# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 13:21:10 2018

@author: yu-hao
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation, ensemble, preprocessing, metrics

h3n2_train = pd.read_csv("E:\\probes_GSE52428matrix_ID_add0.txt",sep='\t',encoding='utf-8')

train=h3n2_train[40:]
train=train.dropna(how='all', axis=1)
train_X=train.drop(['Label','ID'],axis=1)
train_y=train['Label']
test= h3n2_train[:40]
test=test.dropna(how='all', axis=1)
test_y=test["Label"]
test_X=test.drop(['Label','ID'],axis=1)
#h1n1_X = h1n1_train.drop(['A/S','ID'],axis=1)
#h1n1_y = h1n1_train["A/S"]
#h1n1_X, test_X, h1n1_y, test_y = cross_validation.train_test_split(h1n1_X, h1n1_y, test_size = 0.103)#40筆data
forest = ensemble.RandomForestClassifier(n_estimators = 100)
forest_fit = forest.fit(train_X, train_y)
importances=forest.feature_importances_
                #重要度演算法 默認是gini
#print feature importance
std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
indices = np.argsort(importances)[::-1]
print("Feature ranking:")

#for f in range(100):
#    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
sorted_feature_importance = sorted(zip(importances, list(train_X)), reverse=True)

               # 預測
test_y_predicted = forest.predict(test_X)

                # 績效
accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print(accuracy)
f1 = open('E:\\random_slecet100_20200113.txt', 'w', encoding = 'UTF-8')
   
                # 也可使用指定路徑等方式，如： C:\A.txt
                #W會覆寫 a才不會 
for f in range(100):
    f1.write(str(sorted_feature_importance[f])+"\n")
#f1.write()
f1.close

