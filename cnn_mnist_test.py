from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
np.random.seed(10)

(x_Train, y_Train), \
(x_Test, y_Test) = mnist.load_data()

#資料預處理
x_Train4D=x_Train.reshape(x_Train.shape[0],28,28,1).astype('float32')
x_Test4D=x_Test.reshape(x_Test.shape[0],28,28,1).astype('float32')

x_Train4D_normalize = x_Train4D/255
x_Test4D_normalize = x_Test4D/255

y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot = np_utils.to_categorical(y_Test)

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

#CNN模型建立，在MLP前新增
model = Sequential()
#建立卷積層1與池化層1
model.add(Conv2D(filters=16,#建立16個濾鏡filter weight
                 kernel_size=(5,5),
#設定每個濾鏡5X5大小
                 padding='same',	
#此設定讓卷積運算，產生的卷積影像大小不變
                 input_shape=(28,28,1),
#第12維度:代表輸入的影像形狀28X28大小，第3維度:單色灰階影像所以最後維度是1
                 activation='relu'))
#設定ReLU激活函數
model.add(MaxPooling2D(pool_size=(2,2)))
#縮減取樣，將16個28x28影像，縮小為16個14x14影像



#建立卷積層2與池化層2
model.add(Conv2D(filters=36,			#建立36個濾鏡filter weight
                 kernel_size=(5,5),	#設定每個濾鏡5X5大小
                 padding='same',		#此設定讓卷積運算，產生的卷積影像大小不變
                 activation='relu'))	#設定ReLU激活函數
model.add(MaxPooling2D(pool_size=(2,2)))
#縮減取樣，將36個14x14影像，縮小為36個7x7影像
model.add(Dropout(0.25))				
#訓練迭代時，隨機在神經網路中放棄25%的神經元

#建立平坦層
model.add(Flatten())
#將36個7X7影像，轉換為1維向量，長度是36X7X7=1764

#建立隱藏層
model.add(Dense(128,activation='relu'))#共有128個神經元
model.add(Dropout(0.5))#訓練迭代時，隨機在神經網路中放棄50%的神經元，避免overfitting
#建立輸出層
model.add(Dense(10,activation='softmax'))#輸出10個神經元，對應0~9共10個數字，使用softmax激活函數
print(model.summary())
#訓練模型
model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])

train_history=model.fit(x=x_Train4D_normalize,
                        y=y_TrainOneHot,validation_split=0.2,
                        epochs=20,batch_size=100,verbose=2)

#以圖形觀察訓練結果
import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show

show_train_history(train_history,'acc','val_acc')

