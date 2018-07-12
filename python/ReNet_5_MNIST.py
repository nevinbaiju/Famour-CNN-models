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


from tensorflow.examples.tutorials.mnist import input_data
# depricated
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True, reshape = False);


def createLeNet():
    model = Sequential()

    # First convolutional layer with filter = (5*5) and strides = 1
    model.add(Convolution2D(filters=6, kernel_size=(5, 5), input_shape=(28, 28, 1), strides=1))

    # Average pooling is applied.
    model.add(AveragePooling2D(pool_size=(2, 2), strides=1))

    # Second convolutional layer is applied with the same kernel size and the same strides.
    model.add(Convolution2D(filters=6, kernel_size=(5, 5), strides=1))

    # A final average pooling layer is applied before using Fully connected layers but with strides=2
    model.add(AveragePooling2D(pool_size=(2, 2), strides=2))

    # We then use fully connected layers

    model.add(Flatten())
    model.add(Dense(units=120))
    model.add(Dense(units=84))
    model.add(Dense(units=10, activation = 'sigmoid'))

    model.compile(optimizer = 'SGD', loss = 'mean_squared_error', metrics = ['accuracy'])
    print(model.summary())
    
    return model


ReNet = createLeNet()

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])