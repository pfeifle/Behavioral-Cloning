import os
import sys
import csv
from PIL import Image
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from scipy.misc import toimage

from lib import *
from keras.layers.normalization import BatchNormalization


DataLength1 = 8000 # training data provided by Udacity
DataLength2 = 2745 # training data recorded by the student focusing on critical situations

# From the training set provided by Udacity images with a hugh steering angle are added several times
# to the training set. The reason for this is that there is a much less number of images with high steering angles and we want to have a more
# equally distributed training set.
# In addition the left and right images are used for big steering angles.
# The overall training set sums up to DataLength

DataLength = 18824 + 2745# overall data length


#Initialization of training set
X_train = np.zeros((DataLength,36,128,3), dtype=np.float32)
y_train = np.zeros((DataLength), dtype=np.float64)


#cnt for images added to the training set
train_cnt=0

with open('./data/driving_log.csv', 'r') as f:
    csvreader =csv.reader (f,delimiter =',')
    i=0
    for row in csvreader:
        if (i==0):
            print ('ignore first row') # header of csv file
        elif (i<=DataLength1):
            img = Image.open('./data/'+row[0])
            img = cut_top_portion_of_images(img) # image cropped to the interesting area. Function is defined in lib.py
            img = img.resize((round(img.size[0]*scale),round(img.size[1]*scale))) # resized to a factor defined in lib.py
            img_data = np.asarray(img)

            # image added to the training set
            X_train[train_cnt] = img_data
            y_train[train_cnt] =  float(row[3])  
            train_cnt = train_cnt +1

            # training images with high steering values are added several times to the training set, i.e.
            # steering_value between -0.1 and -0.2 or 0.1 and 0.2, it is added one more time
            # steering_value between -0.2 and -0.3 or 0.2 and 0.3, it is added two more times 
            #... 
            # for theses additional added images, the left and right image is also added to the training set.
            # In this case the steering angles are corrected by a value of +/- 0.15
            add_iterations = int(abs(float(row[3])*10) // 1)
            if ((i%1000)==0):
                print (i)
            
            for l in range(add_iterations):
                img=Image.open('./data/'+row[0])
                img =cut_top_portion_of_images(img)
                img =  img.resize((round(img.size[0]*scale),round(img.size[1]*scale)))
                img_data = np.asarray(img)
                X_train[train_cnt] = img_data
                y_train[train_cnt] =  float(row[3])  
                train_cnt = train_cnt +1
                
                img=Image.open('./data/'+row[1]) # left image
                img =cut_top_portion_of_images(img)
                img =  img.resize((round(img.size[0]*scale),round(img.size[1]*scale)))
                img_data = np.asarray(img)
                X_train[train_cnt] = img_data
                y_train[train_cnt] =  float(row[3]) +0.15 # steering values are mapped to integer values
                train_cnt = train_cnt +1

                img=Image.open('./data/'+row[2]) # right image
                img =cut_top_portion_of_images(img)
                img =  img.resize((round(img.size[0]*scale),round(img.size[1]*scale)))
                img_data = np.asarray(img)
                X_train[train_cnt] = img_data
                y_train[train_cnt] =  float(row[3]) -0.15 # steering values are mapped to integer values
                train_cnt = train_cnt +1
          
        else:
            break
        i =i+1
    
# additonally recorded images are added to the training set
with open('../Simulator/data_brokenCurve/driving_log.csv', 'r') as f:
    csvreader =csv.reader (f,delimiter =',')
    i=0
    for row in csvreader:
        if (i==0):
            print ('ignore first row')
        elif (i<=DataLength2):
            img=Image.open(row[0])
            img =cut_top_portion_of_images(img)
            img =  img.resize((round(img.size[0]*0.5),round(img.size[1]*0.5)))
            img_data = np.asarray(img)
            X_train[train_cnt] = img_data
            y_train[train_cnt] =  float(row[3]) 
            train_cnt = train_cnt +1
        else:
            break
        i =i+1

        
print('number of training images: ', train_cnt);



##After reading images and preprocessing....
X_train = np.array(X_train)
#shuffle and split Training Data into Train and Validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

#setting parameters for optimization
batch_size = 128
nb_classes = 1
nb_epoch = 20
X_train = X_train.astype('float32')
X_test = X_val.astype('float32')
#print(X_train.shape[0], 'train samples')
#print(X_val.shape[0], 'test samples')
input_shape = X_train.shape[1:]
#print(input_shape)




#MODEL DEFINITION AND TRAINING 
#it is basically the model proposed by Nvidia, cf.
#with some small modifications in the fully connected layers.
model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))
#Start with 4 Convolutiional Layers to recognize the image
model.add(Convolution2D(24, 5, 5, subsample=(5, 5), border_mode='same', input_shape=input_shape, activation='relu', dim_ordering='tf'))
model.add(Convolution2D(36, 5, 5, border_mode='same', input_shape=input_shape, activation='relu', dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same', dim_ordering='tf'))
model.add(Convolution2D(48, 5, 5, border_mode='same', input_shape=input_shape, activation='relu', dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same', dim_ordering='tf'))
model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=input_shape, activation='relu', dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same', dim_ordering='tf'))
model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=input_shape, activation='relu', dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same', dim_ordering='tf'))
#model.add(Dropout(0.25))
#Flatten the Matrix to a Vector and run 3 RELU Layers
model.add(Flatten())
model.add(Dense(1164, name="hidden1"))
model.add(Activation('relu'))
model.add(Dense(200, name="hidden2"))
model.add(Activation('relu'))
model.add(Dense(50, name="hidden3"))
model.add(Activation('relu'))
model.add(Dense(20, name="hidden4"))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1, name="Steering_Angle"))
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_val, y_val))

 
#SAVE MODEL TO FILES
# serialize model to JSON and save it to a file
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5 and save it to a file
model.save_weights("model.h5")
print("Saved model to disk")


#TEST
#Test the value of some images and print them

with open('./data/driving_log.csv', 'r') as f:
    csvreader =csv.reader (f,delimiter =',')
    i=0
    for row in csvreader:
        if (i==0):
            print ('ignore first row')
        elif (i%25 ==0):
            img = Image.open('./data/'+row[0])
            img = cut_top_portion_of_images(img)
            img =  img.resize((round(img.size[0]*scale),round(img.size[1]*scale)))
            img_data = np.asarray(img)
            transformed_image_array = img_data[None, :, :, :]
            steerings = model.predict(transformed_image_array, batch_size=1)
            print(row[3], " ", steerings[0])
        i=i+1
    print ("end")


