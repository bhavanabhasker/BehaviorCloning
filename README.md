# BehaviorCloning
DNN for self driving car to predict steering angles from the images 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around a track without leaving the road

#### Files included in the repo
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md: summarizing the results

#### How to execute?

The drive.py takes the input from the simulator
The model.h5 is the trained CNN model 
Run the command, 
<pre>
python drive.py model.h5 
</pre>

### Model Architecture and Training Strategy

#### 1. Model architecture:
Model consists of 
1. 5 convolution layers to capture features 
2. RELU activation layers to capture non linear relationships
3. Normalization layers introduced using Keras Batch Normalization layer
4. 3 Fully connected layers at the end to determine the steering angles 
5. Adam Optimizer to minimize the loss 

#### 2. Overfitting 
1. To reduce the overfitting, I have introduced dropout layers. 
2. The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by 
running it through the simulator and ensuring that the vehicle could stay on track.

#### 3. Model parameter tuning 

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
I used a combination of center lane driving, recovering from the left and right sides of the road. 

### Solution Design Approach

My first step was to use a convolution neural network model similar to LeNet Architecture.
I thought this model might be appropriate because the mixture of Convolution and Max Pooling layers are capable of capturing non 
linear relationship in the data. 

In order to gauge how well the model was working, 
I split my image and steering angle data into a training and validation set. I found that my first 
model had a low mean squared error on the training set but a high mean squared error on the validation set. 
This implied that the model was overfitting. 

I modified the model to use Nvidia's Architecture. 
Then I used dropout layers to prevent overfitting. This architecture performed better than the initial one. 

The final step was to run the simulator to see how well the car was driving around track one.
There were a few spots where the vehicle fell off the track;to improve the driving behavior in these cases, I used following 
strategy to extract images:
1. Collect the images from the recoveries
2. Ensure that the car stays on the center of the lane 
3. one lap focusing on driving smoothly around curves

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

