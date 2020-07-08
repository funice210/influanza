# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:16:14 2018

@author: kc
"""

import tensorflow as tf

input1 = tf.placeholder(tf.float32)#type
input2 = tf.placeholder(tf.float32)#type

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[7.],input2:[2.]}))