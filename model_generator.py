import csv
import os
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
import sklearn

file_url='../data/'
####### 0. Build generator + datasets
samples = []
with open(file_url+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    im=mpimg.imread(file_url+batch_sample[i])
                    images.append(im)
                    angle=float(batch_sample[3])
                    if i==1:
                        angle=np.copy(angle)+0.2
                    if i==2:
                        angle=np.copy(angle)-0.2
                    angles.append(angle)
            aug_im,aug_meas=[],[] 
            for image,measurment in zip(images,angles):
                aug_im.append(image)
                aug_meas.append(measurment)
                aug_im.append(cv2.flip(image,1))
                aug_meas.append(measurment*-1.0)
            num_samples = len(aug_im)
            X_train = np.array(aug_im)
            y_train = np.array(aug_meas)
            #yield tuple(sklearn.utils.shuffle(X_train, y_train))
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)
num_train=6*len(train_samples)
num_val=6*len(validation_samples)
imshape=(160,320,3)
imshape2=(160,320,1)
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
#model.add(Conv2D(filters=16,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=32,kernel_size=(5,5),activation='relu',padding='same'))
#model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same'))
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

####### II train & save model
model.compile(optimizer='adam',loss='mse')
model.fit_generator(train_generator, steps_per_epoch=num_train,validation_data=validation_generator,\
                    validation_steps=num_val, epochs=7,verbose=1,\
                    workers=8,use_multiprocessing=True)
model.save('model0.h5')