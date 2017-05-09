import csv
import cv2 
import matplotlib as plt
import numpy as np
import gc
import argparse
import json
from keras.models import Sequential
from keras.layers import Activation,Flatten, Dense ,Lambda, Dropout,SpatialDropout2D 
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D 
from keras.layers import Cropping2D
from keras.layers.normalization import BatchNormalization 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.sparse import *
import pandas as pd 
# Normalize and center the image 
def normalize(image):
 image = (image /255.0) - 0.5
 return image 

# Preprocess the image
def preprocess_image(file_loc, row_val):
  source = row_val.split('/')[-1]
  image = cv2.imread(file_loc + '/IMG/'+ source)
  #image = normalize(image)
  return image 
# Read the Images: Normalize and flip images 
def process_images():
  files = ['lap_1', 'lap_1rev' ]
  for file_loc in files:
     lines = []
     with open(file_loc+'/driving_log.csv') as csvhandler:
       reader = csv.reader(csvhandler)
       for line in reader: 
          lines.append(line)
     images = [] 
     measurements = []
     print("Reading and Normalizing images")
     for line in lines: 
        center_image = preprocess_image(file_loc, line[0])
        steering_center = float(line[3])
        left_image = preprocess_image(file_loc,line[1])
        steering_left = steering_center + 0.2 
        right_image = preprocess_image(file_loc, line[2])
        steering_right = steering_center - 0.2  
        images.extend([center_image, left_image, right_image])
        measurements.extend([steering_center, steering_left, steering_right])
  return images, measurements

# Resize the images  
def process(img):
    import tensorflow as tf
    img = tf.image.resize_images(img, (66, 200))
    return img

# Use Nvidia's DNN Architecture
def nvidia_model():
   row, col, depth = 66,200,3 
   model = Sequential()
   model.add(Lambda(process,input_shape = (160,320,3) ))
   model.add(Lambda(lambda x: x/255.-0.5))
   model.add(Convolution2D(24, 5, 5, border_mode="same", subsample=(2,2), activation="relu"))
   model.add(SpatialDropout2D(0.2))
   model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2,2), activation="relu"))
   model.add(SpatialDropout2D(0.2))
   model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2,2), activation="relu"))
   model.add(SpatialDropout2D(0.2))
   model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="relu"))
   model.add(SpatialDropout2D(0.2))
   model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="relu"))
   model.add(SpatialDropout2D(0.2))
   model.add(Flatten())
   model.add(Dropout(0.5))
   model.add(Dense(100, activation="relu"))
   model.add(Dense(50, activation="relu"))
   model.add(Dense(10, activation="relu"))
   model.add(Dropout(0.5))
   model.add(Dense(1))  
   model.summary()
   # Adam loss optimizer to reduce error 
   model.compile(loss = 'mse' , optimizer = 'adam')
   return model
# Flip Images for augmentation 
def flip_image(images , measurements): 
  aug_features, aug_measures = [], []
  for i in range(len(images)):
    aug_features.append(images[i])
    aug_measures.append(measurements[i])
    flip_image = cv2.flip(images[i], 1) 
    flip_measure = measurements[i]* -1 
    aug_features.append(flip_image) 
    aug_measures.append(flip_measure) 
  aug_features = np.array(aug_features)
  aug_measures = np.array(aug_measures) 
  return aug_features, aug_measures   
  

# Generator function for training 
def train_generator(features, labels, batch_size, num_per_epoch):
   while True: 
        val = min(len(features), num_per_epoch)
        iteration = int(val/ batch_size)
        for i in range(iteration):
           start, end = i*batch_size, (i+1)*batch_size
           aug_features, aug_labels =  flip_image(features[start:end], labels[start:end])
           yield aug_features, aug_labels

# Generator function for  validation 
def val_generator(features, labels, batch_size, num_per_epoch):
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
   images , measurements = process_images()
   X_train = np.array(images)
   y_train = np.array(measurements)
   # Shuffle and split the training data 
   #print(X_train.shape, y_train.shape)
   X_train, y_train = shuffle(X_train, y_train)
   X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=.25, random_state=0)
   print("Building model")
   model = nvidia_model()
   for i in range(args.epoch):
        print(i)
        score = model.fit_generator(
          train_generator(X_train, y_train,args.batch,args.epochsize),nb_epoch=1,samples_per_epoch=len(X_train),
             validation_data=val_generator(X_val,y_val, args.batch, args.epochsize),nb_val_samples=len(X_val),verbose=1)
   model_json = model.to_json() 
   with open('model.json', 'wb') as out:
        json.dump(model_json,out) 
        model.save_weights('model.h5')
        print("Model saved")
