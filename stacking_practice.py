# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:47:01 2019

@author: kc
"""

# Load in our libraries
import pandas as pd
pd.options.mode.chained_assignment = None

titanic = pd.read_csv('E:/titanic.csv')

X = titanic[['Pclass', 'Age', 'Sex']]
y = titanic['Survived']

X['Age'] = X['Age'].fillna(X['Age'].mean())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

