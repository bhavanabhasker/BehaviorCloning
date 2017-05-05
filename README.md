# BehaviorCloning
DNN for self driving car to predict steering angles from the images 

[//]: # (Image References)

[image1]: ./examples/Nvidia_model.png "Model Visualization"
[image2]: ./examples/center.jpg 
[image3]: ./examples/rec_center.jpg "Recovery Image"
[image4]: ./examples/rec_left.jpg "Recovery Image"
[image5]: ./examples/rec_right.jpg "Recovery Image"
[image6]: ./examples/orig.jpg   "Normal Image"
[image7]: ./examples/flipped.jpg "Flipped Image"

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


### Final Model Architecture

The final model architecture  consisted of a convolution neural network with the following layers and layer sizes as shown in the image:

![alt text][image1]

#### Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn not steer off the road. These images show what a recovery looks like starting :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would create more images for training. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

 I then preprocessed this data by following :
 1. Normalizing the images
 2. Cropping the images 
 3. Flipping the images 

I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 . I used an adam optimizer so that manually training the learning rate wasn't necessary.

