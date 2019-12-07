# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:58:24 2019

@author: acani
"""
import os
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Denoise(tf.keras.Model):
    def __init__(self):
	    # reference section 5.2
	    
	    self.input_patch_size = 65
	    self.output_patch_size = 21
	    self.input_channels = 3 # will be up to 27 i believe
	    self.output_channels = 3
	    self.num_conv_layers = 9
	    self.filters = [100, 100, 100, 100, 100, 100, 100, 100, output_patch_size * output_patch_size] # for kpcn, otherwise final layer size is output channels
	    self.kernel_sizes = [5, 5, 5, 5, 5, 5, 5, 5, 5]

	    assert(len(self.filters) == self.num_conv_layers)
	    assert(len(self.kernel_sizes) == self.num_conv_layers)

	    self.kernels = []
	    
	    for i in range(num_conv_layers):
		    activation = 'relu' if i != num_conv_layers - 1 else None
		    self.kernels.append(
				    tf.keras.layers.Conv2d(filters[i], self.kernel_sizes[i], activation=activation) # padding is valid in training, but not when evaluating, keras layes may not be ideal for this

	    assert(len(self.kernels) == self.num_conv_layers)
        
    def call(self, images):
	    
	    # should convolving with kernel happen here or in loss, my intuition is in loss
	    out = images
	    for i in range(self.num_conv_layers):
	    	out = self.kernels(out)
	return out

    def loss(self, denoised, original):
        """
        Computes the loss for a batch of images.

        :param denoised: a 4d matrix (batchsize, width, height, 3) of pixel values representing a batch of image, the output of call
        :param original: a 4d matrix (batchsize, width, height, 3) of pixel values representing a batch of image, the input of call
        :return: loss, a TensorFlow scalar
        """
        denoised = tf.dtypes.cast(denoised, tf.float32)
        original = tf.dtypes.cast(original, tf.float32)

	# model.call() here
        
        return tf.reduce_sum(tf.abs(denoised - original))
    
    def accuracy(self, denoised, original):
        """
        Computes the accuracy for a batch of images.

        :param denoised: a 4d matrix (batchsize, width, height, 3) of pixel values representing a batch of image, the output of call
        :param original: a 4d matrix (batchsize, width, height, 3) of pixel values representing a batch of image, the input of call
        :return: accuracy, a TensorFlow scalar
        """
        denoised = tf.dtypes.cast(denoised, tf.float32)
        original = tf.dtypes.cast(original, tf.float32)
        
        '''
        for each image in the batch, average the differences between each color channel
        then average these across each pixel of each image 
        '''
        return tf.reduce_mean(tf.reduce_mean(tf.abs(denoised - original), 3))
        
