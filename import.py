# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:26:56 2017

@author: kc
"""
import tensorflow as tf
x=tf.constant(1,name='x')
y=tf.Variable(x+9,name='y')
model=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(model)
    print(sess.run(y))