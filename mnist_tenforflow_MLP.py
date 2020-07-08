# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 13:27:40 2018

@author: kc
"""

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

#建立layer函數
def layer(output_dim,input_dim,inputs,activation=None):
    W=tf.Variable(tf.random_normal([input_dim,output_dim]))
    b=tf.Variable(tf.random_normal([1,output_dim]))
    XWb=tf.matmul(inputs,W)+b
    if activation is None:
        outputs=XWb
    else:
        outputs=activation(XWb)
    return outputs

#建立輸入層x
x=tf.placeholder("float",[None,784])
#1維設定為None，筆數不固定，2維設定為784，數字影像像素是784

#建立隱藏層h1
h1=layer(output_dim=500,input_dim=784,inputs=x,activation=tf.nn.relu)
#建立隱藏層神經元個數256，輸入神經元也就是數字影像像素784，x為輸入層，定義激活函數tf.nn.relu
h2=layer(output_dim=500,input_dim=500,inputs=h1,activation=tf.nn.relu)
#建立輸出層y
y_predict=layer(output_dim=10,input_dim=500,inputs=h2,activation=None)
#建立輸出層神經元個數10，神經元個數256，h1隱藏層，不須激活函數

#定義訓練方式
#建立訓練資料lebel真實值的placeholder
y_label=tf.placeholder("float",[None,10])

#定義loss function，使用cross_entropy
loss_function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict,labels=y_label))

#定義optimizer
optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)

#計算每筆資料是否預測正確
correct_prediction=tf.equal(tf.argmax(y_label,1),tf.argmax(y_predict,1))
#將計算預測正確結果平均
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

#進行訓練
trainEpochs=20
batchSize=100
totalBatchs=int(mnist.train.num_examples/batchSize)
loss_list=[];epoch_list=[];accuracy_list=[]
from time import time
startTime=time()

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(trainEpochs):
    for i in range(totalBatchs):
        batch_x,batch_y=mnist.train.next_batch(batchSize)
        sess.run(optimizer,feed_dict={x: batch_x, y_label:batch_y})
    loss,acc=sess.run([loss_function,accuracy],feed_dict={x:mnist.validation.images,y_label:mnist.validation.labels})
    
    epoch_list.append(epoch);
    loss_list.append(loss)
    accuracy_list.append(acc)
    print("Train Epoch:",'%02d' % (epoch+1),"Loss=","{:.9f}".format(loss), " Accuracy=",acc)
   
duration=time()-startTime


print("Train Finished takes:",duration)   
    
#畫出loss誤差結果
import matplotlib.pyplot as plt
fig=plt.gcf()
fig.set_size_inches(4,2)
plt.plot(epoch_list,loss_list,label='loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'],loc='upper left')

#畫出accuracy準確率結果
#fig=plt.gcf()
#fig.set_size_inches(4,2)
#plt.ylim(0.8,1)
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend()
#plt.show()

#評估模型準確率
print("Accuracy:",sess.run(accuracy,feed_dict={x:mnist.test.images,y_label:mnist.test.labels}))