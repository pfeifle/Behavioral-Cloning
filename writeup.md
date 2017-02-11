# Behavrioal Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[video1]: ./demo.mp4 "demo drive"
[image1]: ./images/lane_center.jpg "Center Lane Driving"
[image2]: ./images/recovery1.jpg "Recovery Image 1"
[image3]: ./images/recovery2.jpg "Recovery Image 2"
[image4]: ./images/recovery3.jpg "Recovery Image 3"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```
A demo of the "self-driving car" can be seen here: [video1]

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed
The model is basically the one described by [Nvidia](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). The model consists of 6 convolutional layers with threee 5x5 kernels and two 3x3 kernels followed by 4 dense layers (cf. model.py ines 140-165)

The model includes RELU layers to introduce nonlinearity (code line 156, 158,160, 162), and the data is normalized in the model using Keras BatchNormalization (code line 141). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 163). 

The model was trained on two different data sets to ensure that the model was not overfitting (code line 38-92 for the first data set and 95-111 for the second). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 166).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.
#####4.1. Udacity Training Set
The images contained in the Udacity training set were added to the test set. As this set contains a lot of images with small steering angle values, images with higher steering values were added several times. The higher the steering value the more often the image was added. In addition, the left and right images were added to the training set as well with adapted steering angles (+/- 0.15 was added to the steering value of the center image). By doing this, a more or less equal distribution of the steering angles within the training set was achieved.

#####4.2 Self-Recorded Training 
In addition, to the Udacity pre-recorded training data an own training set was recorded which contained difficult curves and some situations where the car was standing at the wall and went back on track.


###Model Architecture and Training Strategy

####1. Solution Design Approach

First, I tried the model described in "traffic sign classification with Keras". This helped to get familiar with Keras. I used the center images of the udacit training set and tried to see whether the trained network predicts some of the training set images correctly. Unfortunatley, this was not the case. I tried a lot things, e.g. changing kernel sizes of convolutional networks,the number of convolutional layers, the number of dense layers, the activation functions, but nothing helped. I even tried to have several output neurons and trained the network to predict "discrete" steering angles such as -0.9, -0.8, ..., -0.1,0,0.1, ..., 0.9, i.e. similar to what we did during the traffic sign classification. But it didn't helped. I never managed to pass the first curve and the predicted steering angles were more or less random.

I tried then the Nvidia network as described in [Nvidia](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
but it didn't helped either. I then concentrated on image pre-processing and cropped the images by removing 50 pixles from the top and 20 from the bottom. This approach in combination with the Nvidia network started to work. I also reduced the cropped images by a factor of 0.4 and then things started to work.
 
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I used 20% for validation. In order to avoid overfitting, I added a dropout layer right before the final output layer. 

As I was still struggling with the sharp curves, I added images with high steering values more frequently to the training set. By doing this, the overall distribution of steering values got more aligned. In addition, I added the left and right images from Udacity with some small correction values.

In order not to overemphasize the high steering values, I recorded a few onw tracks focusing on recovery situation, e.g. getting from the wall back to track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
One remark: as I am working on a very slow computer without gpu support, I struggled with high throttle values and high resolution simulators. The car is running several times (> 20) around the track for throttle values of 0.2 (or 0.1) with the fast rendering.

####2. Final Model Architecture

The final model architecture (model.py lines 140-164) consisted of a convolution neural network with the following layers and layer sizes 
 
Layer (type) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Output&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Shape&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          Param Nrs&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     Connected to
____________________________________________________________________________________________________
batchnormalization_1 (BatchNorma  &nbsp;&nbsp; (None, 36, 128, 3) &nbsp;&nbsp;   12 &nbsp;&nbsp;         batchnormalization_input_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  &nbsp;&nbsp;  (None, 8, 26, 24)  &nbsp;&nbsp;   1824   &nbsp;&nbsp;     batchnormalization_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D) &nbsp;&nbsp;   (None, 8, 26, 36)  &nbsp;&nbsp;   21636    &nbsp;&nbsp;   convolution2d_1[0][0]
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)   &nbsp;&nbsp;   (None, 4, 13, 36) &nbsp;&nbsp;    0    &nbsp;&nbsp;       convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D) &nbsp;&nbsp;   (None, 4, 13, 48)   &nbsp;&nbsp;  43248  &nbsp;&nbsp;     maxpooling2d_1[0][0]
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    &nbsp;&nbsp;  (None, 2, 7, 48)  &nbsp;&nbsp;    0     &nbsp;&nbsp;      convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  &nbsp;&nbsp;  (None, 2, 7, 64)   &nbsp;&nbsp;   27712  &nbsp;&nbsp;     maxpooling2d_2[0][0]
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    &nbsp;&nbsp;  (None, 1, 4, 64)   &nbsp;&nbsp;   0    &nbsp;&nbsp;       convolution2d_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  &nbsp;&nbsp;  (None, 1, 4, 64) &nbsp;&nbsp;     36928 &nbsp;&nbsp;      maxpooling2d_3[0][0]
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)  &nbsp;&nbsp;    (None, 1, 2, 64) &nbsp;&nbsp;     0     &nbsp;&nbsp;      convolution2d_5[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)            &nbsp;&nbsp;    (None, 128)     &nbsp;&nbsp;      0    &nbsp;&nbsp;       maxpooling2d_4[0][0]
____________________________________________________________________________________________________
hidden1 (Dense)                &nbsp;&nbsp;    (None, 1164)    &nbsp;&nbsp;      150156&nbsp;&nbsp;      flatten_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)     &nbsp;&nbsp;     (None, 1164)    &nbsp;&nbsp;      0     &nbsp;&nbsp;      hidden1[0][0]
____________________________________________________________________________________________________
hidden2 (Dense)                    (None, 200)           233000      activation_1[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)          (None, 200)           0           hidden2[0][0]
____________________________________________________________________________________________________
hidden3 (Dense)                    (None, 50)            10050       activation_2[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)          (None, 50)            0           hidden3[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)                (None, 50)            0           activation_3[0][0]
____________________________________________________________________________________________________
hidden4 (Dense)                    (None, 20)            1020        dropout_1[0][0]
____________________________________________________________________________________________________
activation_4 (Activation)          (None, 20)            0           hidden4[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)                (None, 20)            0           activation_4[0][0]
____________________________________________________________________________________________________
Steering_Angle (Dense)             (None, 1)             21          dropout_2[0][0]
____________________________________________________________________________________________________
Total params: 525,607
Trainable params: 525,601
Non-trainable params: 6

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to the center of the lane. The three images below show such a recovery, i.e. steering back to the  of the lane: 

![alt text][image2]


![alt text][image3]


![alt text][image4]

Then I repeated this process on track two in order to get more data points.

I added the 2745 recorded imagest to the Udacity training set. Overall, I used 21,569 images for the training. The Udacity training images and the self-recorded images were preprocessed in the same way. The training of the model was done without gpu support. 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20. After20 epochs the validation loss did not further decrease. I used an adam optimizer so that manually training the learning rate wasn't necessary.
