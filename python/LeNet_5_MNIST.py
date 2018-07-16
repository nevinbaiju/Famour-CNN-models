#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 19:01:58 2018

@author: Nevin Baiju
@email : nevinbaiju@gmail.com

Implementation of LeNet-5 using keras
"""

from keras.models import Sequential

from keras.layers import Convolution2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense

import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from keras.datasets import mnist

def process_batch(batch):
	"""
	This function takes in a batch of single channel images as array and then resizes
	according to the requirements and returns the processed batch.
	
	Parameters
	----------
	batch: numpy array 
		   Batch of images given for processing as (batch_size, img_width, img_height, channel).
	Returns
	-------
	processed_batch: numpy array
		  The processed batch of images.
	
	"""
	processed_batch = []
	for img_arr in batch:
		img = Image.fromarray(img_arr)
		img = img.resize((32, 32), Image.ANTIALIAS)
		img_arr = np.reshape(np.array(img), newshape=(32, 32, 1))
		processed_batch.append(img_arr)
		
	return np.array(processed_batch)
		
def create_genrator(data, batch_size):
	"""
	This function is used for creating a generator object that iterates over the total input data after processing it.
	The x data is resized.
	The y data is one hot encoded.
	
	Parameters
	----------
	data              :tuple 
					   Input data in the form of x and y values as a tuple.
	batch_size        :integer
					   The number of batches to which the original data size needs to be split.
	Returns
	-------
	data_generator    :generator object
					   The generator object that supplies in processed x values and the respective y label.
	
	"""
	x = data[0]
	y = data[1]
	
	one_h = OneHotEncoder()
	arr = one_h.fit_transform(y.reshape(-1, 1))
	y = np.array(arr.toarray())
	
	index = -1
			 
	while(1):
		index = index + batch_size
		processed_batch_x = process_batch(x[index:index+batch_size])
		processed_batch_y = y[index:index+batch_size]
		
		if not(index <= len(x)):
			print(index, sep='\r', end='\r')
			index = -1
		else:
			 yield [processed_batch_x, processed_batch_y]

def createLeNet():
	"""
	Function for adding layers to the neural network.
	
	Parameters
	----------
	None
	Returns
	-------
	None
	"""
	model = Sequential()

	# First convolutional layer with filter = (5*5) and strides = 1
	model.add(Convolution2D(filters=6, kernel_size=(5, 5), input_shape=(32, 32, 1), strides=1, activation='relu'))

	# Average pooling is applied.
	model.add(AveragePooling2D(pool_size=(2, 2), strides=2))

	# Second convolutional layer is applied with the same kernel size and the same strides.
	model.add(Convolution2D(filters=16, kernel_size=(5, 5), strides=1, activation='relu'))

	# A final average pooling layer is applied before using Fully connected layers but with strides=2
	model.add(AveragePooling2D(pool_size=(2, 2), strides=2))

	# We then use fully connected layers

	model.add(Flatten())
	model.add(Dense(units=120))
	model.add(Dense(units=84))
	model.add(Dense(units=10, activation = 'softmax'))

	model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
	print(model.summary())
	
	return model

def train():
	"""
	Function for training the neural network.
	
	Parameters
	----------
	None
	Returns
	-------
	None
	"""

	LeNet = createLeNet()

	train, test = mnist.load_data()

	train_gen = create_genrator(train, 32)
	history = LeNet.fit_generator(generator=train_gen, 
								  steps_per_epoch=60000/32, 
								  epochs=15)


if __name__ == '__main__':
	train()