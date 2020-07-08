# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:37:18 2019

@author: yu-hao
"""
import numpy
import pandas as pd
from sklearn import preprocessing
from sklearn import cross_validation, svm, metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
all_df = pd.read_csv("F:/AUC.txt",sep='\t',encoding='utf-8')

l = all_df.values[:,0]
p_pc = all_df.values[:,1]
#p_b=all_df.values[:,2]
#roc
fpr, tpr, thresholds = metrics.roc_curve(l,p_pc)
auc_roc = metrics.auc(fpr, tpr)
#pr

precision, recall, thresholds = precision_recall_curve(l,p_pc)
auc_pr = metrics.auc(recall, precision)

print(auc_roc,auc_pr)
