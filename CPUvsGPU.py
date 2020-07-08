# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:00:41 2017

@author: kc
"""

import tensorflow as tf
import time

def performanceTest(device_name,size):
    with tf.device(device_name):
        W = tf.random_normal([size, size],name='W')
        X = tf.random_normal([size, size],name='X')
        mul = tf.matmul(W, X,name='mul')
        sum_result = tf.reduce_sum(mul,name='sum')
        
    startTime =  time.time()
    tfconfig=tf.ConfigProto(log_device_placement=True)
    with tf.Session(config=tfconfig) as sess:
            result = sess.run(sum_result)
    takeTimes=time.time() - startTime
    print(device_name," size=",size,"Time:",takeTimes)
    
    gpu_set=[];cpu_set=[];i_set=[]  
    for i in range(0,5001,500):
        g=performanceTest("/gpu:0",i)
        c=performanceTest("/cpu:0",i)
        gpu_set.append(g);cpu_set.append(c);i_set.append(i)
    
    import matplotlib.pyplot as plt
    fig = plt.gcf()
    fig.set_size_inches(6,4)
    plt.plot(i_set, gpu_set, label = 'gpu')
    plt.plot(i_set, cpu_set, label = 'cpu')
    plt.legend()