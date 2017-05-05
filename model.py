import csv
import cv2 
import matplotlib as plt
import numpy as np
import gc
import argparse
from keras.models import Sequential
from keras.layers import Activation,Flatten, Dense ,Lambda, Dropout 
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D 
from keras.layers import Cropping2D
from keras.layers.normalization import BatchNormalization 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Normalize and center the image 
def normalize(image):
 image = (image /255.0) - 0.5
 return image 

# Preprocess the image
def preprocess_image(file_loc, row_val):
  source = row_val.split('/')[-1]
  image = cv2.imread(file_loc + '/IMG/'+ source)
  image = normalize(image)
  return image 

# Read the Images: Normalize and flip images 
def process_images():
  files = ['training', 'reverse_track' ]
  for file_loc in files:
     lines = []
     with open(file_loc+'/driving_log.csv') as csvhandler:
       reader = csv.reader(csvhandler)
       for line in reader: 
          lines.append(line)
     images = [] 
     measurements = []
     print("Reading and Normalizing images")
     #Record the measurements from center, left and right cameras  
     for line in lines: 
        center_image = preprocess_image(file_loc, line[0])
        steering_center = float(line[3])
        left_image = preprocess_image(file_loc,line[1])
        # Add correction factor 
        steering_left = steering_center + 0.2 
        right_image = preprocess_image(file_loc, line[2])
        steering_right = steering_center - 0.2  
        images.extend([center_image, left_image, right_image])
        measurements.extend([steering_center, steering_left, steering_right])
  augmented_images, augmented_measurements = [], []
  # Flip the image using opencv
  for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
  print("Saving images")
  np.save('Images', augmented_images)
  np.save('Measurements', augmented_measurements)
  del augmented_images, augmented_measurements, images, measurements 
# Use Nvidia's DNN Architecture
def nvidia_model():
   row, col, depth = 66,200,3 
   model = Sequential()
   # Add keras layer for cropping the images 
   model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
   model.add(BatchNormalization(input_shape = (160,320,3), epsilon=1e-6, weights=None))
   model.add(Convolution2D(24,5,5, subsample=(2,2), border_mode = 'valid', activation = "relu"))
   model.add(Convolution2D(36,5,5, subsample=(2,2), border_mode = 'valid' , activation = "relu"))
   model.add(Convolution2D(48,5,5, subsample=(2,2), border_mode = 'valid', activation = "relu"))
   model.add(Convolution2D(64,3,3, subsample=(1,1), border_mode = 'valid', activation = "relu"))
   model.add(Convolution2D(64,3,3, subsample=(1,1), border_mode = 'valid', activation = "relu"))
   model.add(Flatten())
   model.add(Dropout(0.5))
   model.add(Activation('relu'))
   model.add(Dense(100))
   model.add(Activation('relu'))
   model.add(Dense(50))
   model.add(Activation('relu'))
   model.add(Dense(10))
   model.add(Activation('relu')) 
   model.add(Dense(1))
   # Adam loss optimizer to reduce error 
   model.compile(loss = 'mse' , optimizer = 'adam')
   return model
# Generator functions for training and validation 
def generator(features, labels, batch_size, num_per_epoch):
    while True:
        val = min(len(features), num_per_epoch)
        iteration = int(val/ batch_size)
        for i in range(iteration):
           start, end = i*batch_size, (i+1)*batch_size 
           yield features[start:end],labels[start:end]
if __name__ == '__main__':
   # Command Line Arguments for epoch and batch 
   parser = argparse.ArgumentParser(description='Model to train steering angles')
   parser.add_argument('--batch', type=int, default=128, help='Batch size.')
   parser.add_argument('--epoch', type=int, default=5, help='Number of epochs.')   
   parser.add_argument('--epochsize', type=int, default=43394, help='How many images per epoch.')   
   parser.set_defaults(skipvalidate=False)
   parser.set_defaults(loadweights=False)
   args = parser.parse_args()
   print("Processing Images")
   process_images()
   X_train = np.load('Images.npy')
   y_train = np.load('Measurements.npy')
   # Shuffle and split the training data 
   X_train, y_train = shuffle(X_train, y_train)
   X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=.1, random_state=0)
   print("Building model")
   model = nvidia_model()
   for i in range(args.epoch):
        print(i)
        score = model.fit_generator(
          generator(X_train, y_train,args.batch,args.epochsize),nb_epoch=1,samples_per_epoch=len(X_train),
             validation_data=generator(X_val,y_val, args.batch, args.epochsize),nb_val_samples=len(X_val),verbose=1)
   model.save('model.h5')

