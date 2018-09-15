import csv
import numpy as np
import cv2
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.models import Model
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D,Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras import regularizers

file_url='../data/'

####### 0. Build data
with open(file_url+'driving_log.csv') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    data = list(csv_reader)
    data=np.array(data[1:])

images,measurements=[],[]
for img in data:
    for i in range(3):
        im=mpimg.imread(file_url+img[i])
        images.append(im)
        angle=float(img[3])
        if i==1:
            angle=angle+0.2
        if i==2:
            angle=angle-0.2
        measurements.append(angle)

aug_im,aug_meas=[],[] 
for image,measurment in zip(images,measurements):
    aug_im.append(image)
    aug_meas.append(measurment)
    aug_im.append(cv2.flip(image,1))
    aug_meas.append(measurment*-1.0)
Xtrain=np.array(aug_im)
ytrain=np.copy(aug_meas)

ntrain=Xtrain.shape[0]
imshape=Xtrain.shape[1:]
imshape2=np.copy(imshape)
imshape2[-1]=1
imshape2=tuple(imshape2)

####### I. Create model
model = Sequential()
### 1 CNN part
model.add(Lambda(lambda x:x/127.5-1,input_shape=imshape))
def converter(im):
    import tensorflow as tf
    return tf.image.rgb_to_grayscale(im)
model.add(Lambda(converter,output_shape=imshape2))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(filters=16,kernel_size=(5,5),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=32,kernel_size=(5,5),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
model.add(Dropout(0.5))
### 2 FC part
model.add(Flatten()) #L1 flatten input
model.add(Dense(128)) #L2 FC -> Relu
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64)) #L2 FC -> Relu #L3 FC -> Relu
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1)) #L3 FC for regression
# model.add(Conv2D(filters=24,kernel_size=(5,5), strides=2, activation='relu'))
# model.add(Conv2D(filters=36,kernel_size=(5,5), strides=2, activation='relu'))
# model.add(Conv2D(filters=48,kernel_size=(5,5), strides=2, activation='relu'))
# model.add(Conv2D(filters=64,kernel_size=(3,3), activation='relu'))
# model.add(Conv2D(filters=64,kernel_size=(3,3), activation='relu'))
# model.add(Flatten())
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(1))

####### II train & save model
model.compile(optimizer='adam',loss='mse')
model.fit(Xtrain,ytrain,batch_size=48,shuffle=True,validation_split=0.2,epochs=10)
model.save('model.h5')
