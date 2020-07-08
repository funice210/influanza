# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:02:12 2018

@author: kc
"""

import tensorflow as tf
import numpy as np

#create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3

###create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -0.1, 1.0))#隨機生成參數變量
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))#預測y與實際y的差別
optimizer = tf.train.GradientDescentOptimizer(0.5)#優化器減少誤差 提升準確度 
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()#初始化結構
###create tensorflow structure end ###

sess = tf.Session()
sess.run(init)#啟動

for step in range(201):
    sess.run(train)
    if step % 20 ==0:
        print(step, sess.run(Weights), sess.run(biases))