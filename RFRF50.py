# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 18:57:46 2018

@author: kc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation, ensemble, preprocessing, metrics

i=0
while i<100: 
    h3n2_train = pd.read_csv('E:\\50RFCNN\\001RF_50_out.txt',sep = '\t',encoding = 'utf-8')
    
    label_encoder = preprocessing.LabelEncoder()

    h3n2_X = pd.DataFrame([
        h3n2_train["213988_s_at"],
        h3n2_train["201230_s_at"],
        h3n2_train["210592_s_at"],
        h3n2_train["202672_s_at"],
        h3n2_train["209906_at"],
        h3n2_train["202688_at"],
        h3n2_train["212206_s_at"],
        h3n2_train["212185_x_at"],
        h3n2_train["217964_at"],
        h3n2_train["212201_at"],
        h3n2_train["206513_at"],
        h3n2_train["204142_at"],
        h3n2_train["208751_at"],
        h3n2_train["218429_s_at"],
        h3n2_train["205660_at"],
        h3n2_train["209593_s_at"],
        h3n2_train["219684_at"],
        h3n2_train["218505_at"],
        h3n2_train["221050_s_at"],
        h3n2_train["35254_at"],
        h3n2_train["205698_s_at"],
        h3n2_train["208012_x_at"],
        h3n2_train["204972_at"],
        h3n2_train["208052_x_at"],
        h3n2_train["206461_x_at"],
        h3n2_train["202269_x_at"],
        h3n2_train["213851_at"],
        h3n2_train["217933_s_at"],
        h3n2_train["218280_x_at"],
        h3n2_train["205099_s_at"],
        h3n2_train["205241_at"],
        h3n2_train["203582_s_at"],
        h3n2_train["AFFX-HUMISGF3A/M97935_3_at"],
        h3n2_train["202380_s_at"],
        h3n2_train["202446_s_at"],
        h3n2_train["219627_at"],
        h3n2_train["221766_s_at"],
        h3n2_train["206133_at"],
        h3n2_train["211889_x_at"],
        h3n2_train["209093_s_at"],
        h3n2_train["209059_s_at"],
        h3n2_train["218559_s_at"],
        h3n2_train["203922_s_at"],
        h3n2_train["219716_at"],
        h3n2_train["204143_s_at"],
        h3n2_train["213294_at"],
        h3n2_train["208087_s_at"],
        h3n2_train["212681_at"],
        h3n2_train["206491_s_at"],
        h3n2_train["204711_at"]]).T

  
    h3n2_y = h3n2_train["Label"]
    h3n2_X, test_X, h3n2_y, test_y = cross_validation.train_test_split(h3n2_X, h3n2_y, test_size = 0.2)
    forest = ensemble.RandomForestClassifier(n_estimators = 100)
    forest_fit = forest.fit(h3n2_X, h3n2_y)
    importances=forest.feature_importances_
    #print feature importance
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]
    #print("Feature ranking:")
    #
    #for f in range(h1n1_X.shape[1]):
    #    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    #
    # # Plot the feature importances of the forest
    #plt.figure()
    #plt.title("Feature importances")
    #plt.bar(range(h1n1_X.shape[1]), importances[indices],
    #       color="r", yerr=std[indices], align="center")
    #plt.xticks(range(h1n1_X.shape[1]), indices)
    #plt.xlim([-1, h1n1_X.shape[1]])
    #plt.show()
    
    # 預測
    test_y_predicted = forest.predict(test_X)
    
    # 績效
    accuracy = metrics.accuracy_score(test_y, test_y_predicted)
    print(accuracy)
    #f1 = open('C:/Users/yu-hao/Desktop/H1N1/random_forest_predicted.txt', 'a', encoding = 'UTF-8')
    #f2 = open('C:/Users/yu-hao/Desktop/H1N1/random_forest_real.txt', 'a', encoding = 'UTF-8')    
    ## 也可使用指定路徑等方式，如： C:\A.txt
    ##W會覆寫 a才不會 
    #
    #for p in test_y_predicted:
    #    f1.write("%d\n" %(p))
    #
    #for o in test_y:
    #    f2.write("%d\n" %(o))
    #
    ##f.write("%.3f\n" % (accuracy))
    #f1.close
    #f2.close    
    f1 = open('E:\Research\prediction100次結果\RFRF\H3N2_prediction_100次_RFRF50.txt', 'a', encoding = 'UTF-8')
       
    # 也可使用指定路徑等方式，如： C:\A.txt
    #W會覆寫 a才不會 
    test = test_y.tolist()
    TP=0
    TN=0
    FP=0
    FN=0
    for j in range(test_y_predicted.size):
        if(test_y_predicted[j]==1 and test_y_predicted[j]==test[j]):
            TP=TP+1
        else:
            TP=TP+0
        
        if(test_y_predicted[j]==0 and test_y_predicted[j]==test[j]):
            TN=TN+1
        else:
            TN=TN+0
        
        if(test[j]==0 and test_y_predicted[j]==1):
            FP=FP+1
        else:
            FP=FP+0
        
        if(test[j]==1 and test_y_predicted[j]==0):
            FN=FN+1
        else:
            FN=FN+0
    print(TP,TN,FP,FN)
    f1.write(str(TP)+"\t"+str(TN)+"\t"+str(FP)+"\t"+str(FN)+"\n")
    
    
    #f.write("%.3f\n" % (accuracy))
    f1.close
    i+=1
    print(i)