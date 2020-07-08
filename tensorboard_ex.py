# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:24:59 2018

@author: kc
"""
import tensorflow as tf
width=tf.placeholder("int32",name='width')
height=tf.placeholder("int32",name='height')
area=tf.multiply(width,height,name='area')

with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    print('area=',sess.run(area,feed_dict={width:6,height:8}))
    
tf.summary.merge_all()
train_writer=tf.summary.FileWriter('log/area',sess.graph)
