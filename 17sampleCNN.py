from keras.utils import np_utils
import numpy as np
np.random.seed(10)
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
#from sklearn import preprocessing



#(x_Train, y_Train), (x_Test, y_Test) = mnist.load_data()

train_data = np.stack([np.array(pd.read_csv("E:/data/train_set/" + str(index) + ".txt" ,sep = '\t',encoding = 'utf-8'))/36000.0 for index in range(1, 18, 1)])
train_data=train_data.reshape(train_data.shape[0],22277,16,1)

"""for index in range(1, 18, 1):
    traindata = pd.read_csv("E:/data/train_set/" + str(index) + ".txt" ,sep = '\t',encoding = 'utf-8') # 讀取資料
    nparr = np.array(traindata) # 轉成np array
    nparr = nparr / 36000.0"""
    

traincsv = pd.read_csv('E:/data/train_set/label.csv', 'r', encoding = 'utf8')


label = np_utils.to_categorical(traincsv)


#train_data=(train_data.shape[0],22277,16,1)
#資料預處理
#x_Train4D=x_Train.reshape(x_Train.shape[0],28,28,1).astype('float32')
#x_Test4D=x_Test.reshape(x_Test.shape[0],28,28,1).astype('float32')

#x_Train4D_normalize = x_Train4D/255
#x_Test4D_normalize = x_Test4D/255

#y_TrainOneHot = np_utils.to_categorical(y_Train)
#y_TestOneHot = np_utils.to_categorical(y_Test)

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
#from keras.callbacks import EarlyStopping, TensorBoard


#CNN模型建立，在MLP前新增
model = Sequential()
#建立卷積層1與池化層1
model.add(Conv2D(filters=16,#建立16個濾鏡filter weight
                 kernel_size=(5,5),
#設定每個濾鏡5X5大小
                 padding='same',	
#此設定讓卷積運算，產生的卷積影像大小不變
                 input_shape=(22277,16,1),
#第12維度:代表輸入的影像形狀28X28大小，第3維度:單色灰階影像所以最後維度是1
                 activation='relu'))
#設定ReLU激活函數
model.add(MaxPooling2D(pool_size=(2,2)))
#縮減取樣，將16個28x28影像，縮小為16個14x14影像



#建立卷積層2與池化層2
model.add(Conv2D(filters=32,		#建立36個濾鏡filter weight
                 kernel_size=(5,5),	#設定每個濾鏡5X5大小
                 padding='same',		#此設定讓卷積運算，產生的卷積影像大小不變
                 activation='relu'))	#設定ReLU激活函數
model.add(MaxPooling2D(pool_size=(2,2)))
#縮減取樣，將36個14x14影像，縮小為36個7x7影像
model.add(Dropout(0.25))				
#訓練迭代時，隨機在神經網路中放棄25%的神經元

model.add(Conv2D(filters=64,		#建立36個濾鏡filter weight
                 kernel_size=(5,5),	#設定每個濾鏡5X5大小
                 padding='same',		#此設定讓卷積運算，產生的卷積影像大小不變
                 activation='relu'))	#設定ReLU激活函數
model.add(MaxPooling2D(pool_size=(2,2)))
#縮減取樣，將36個14x14影像，縮小為36個7x7影像
model.add(Dropout(0.25))		
#建立平坦層
model.add(Flatten())
#將36個7X7影像，轉換為1維向量，長度是36X7X7=1764

#建立隱藏層
model.add(Dense(100,activation='relu'))#共有128個神經元
model.add(Dropout(0.25))#訓練迭代時，隨機在神經網路中放棄50%的神經元，避免overfitting
model.add(Dense(100,activation='relu'))#共有128個神經元
model.add(Dropout(0.25))#訓練迭代時，隨機在神經網路中放棄50%的神經元，避免overfitting


#建立輸出層
model.add(Dense(2,activation='softmax'))#輸出2個神經元，對應0,1共2個數字，使用softmax激活函數
print(model.summary())
#訓練模型
model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])

train_history=model.fit(x=train_data,
                        y=label,validation_split=0.2,
                        epochs=50,batch_size=17,verbose=2)

scores=model.evaluate(x=train_data,y=label)
accuracy = scores[1]
print(model.summary())
print('accuracy=',accuracy)
#以圖形觀察訓練結果
import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

show_train_history(train_history,'acc','val_acc')


def show_train_history1(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['train_loss','validation_loss',],loc='upper left')
    plt.show()

show_train_history1(train_history,'loss','val_loss')
#del(model)


filepath="E:/data/model/model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_digit6_acc', verbose=1, save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_digit6_acc', patience=5, verbose=1, mode='auto')
tensorBoard = TensorBoard(log_dir = "./logs", histogram_freq = 1)
callbacks_list = [checkpoint, earlystop, tensorBoard]