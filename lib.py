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
from PIL import Image

scale = 0.4

#Pre-processing Images
#Cutting the image to the section, that holds the road information
def cut_top_portion_of_images(image):
    width = image.size[0]
    height = image.size[1]
    return image.crop(( 0, 20, width, height -50 ))
    
#Converting the RGB Image to an HLS Image
# currently not used
def grayscale(img):
    gScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    image_Expanded = gScale[:,:,np.newaxis]
    return image_Expanded

#Normalizing the input Image
# currently not used
def normalize(image_data):
    a = 0.01
    b = 0.99
    color_min = 0.0
    color_max = 255.0
    return a + ( ( (image_data - color_min) * (b - a) )/(color_max - color_min))

