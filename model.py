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
        super(Denoise, self).__init__()

        self.input_patch_size = 65
        self.output_patch_size = 21
        self.input_channels = 3 # will be up to 27 i believe
        self.output_channels = 3
        self.num_conv_layers = 9
        self.filters = [100, 100, 100, 100, 100, 100, 100, 100, self.output_patch_size * self.output_patch_size] # for kpcn, otherwise final layer size is output channels
        self.kernel_sizes = [5, 5, 5, 5, 5, 5, 5, 5, 5]

        assert len(self.filters) == self.num_conv_layers
        assert len(self.kernel_sizes) == self.num_conv_layers

        self.diffuse_kernels = []
        #self.specular_kernels = []

        for i in range(self.num_conv_layers):
            activation = 'relu' if i != self.num_conv_layers - 1 else None
            self.diffuse_kernels.append(tf.keras.layers.Conv2D(self.filters[i], self.kernel_sizes[i], padding='same', activation=activation)) # padding is valid in training, but not when evaluating, keras layers may not be ideal for this
            #self.specular_kernels.append(tf.keras.layers.Conv2D(self.filters[i], self.kernel_sizes[i], padding='same', activation=activation))

        self.diffuse_kernels.append(tf.keras.layers.Conv2D(3, 21, padding='same', activation=None))
        #self.specular_kernels.append(tf.keras.layers.Conv2D(3, 21, padding='same', activation=None))
        assert len(self.diffuse_kernels) == self.num_conv_layers + 1
        #assert len(self.specular_kernels) == self.num_conv_layers + 1

    def call(self, diffuse_images, specular_images):

        # should convolving with kernel happen here or in loss, my intuition is in loss
        diff_out = tf.dtypes.cast(diffuse_images, dtype=tf.float32)
        spec_out = tf.dtypes.cast(specular_images, dtype=tf.float32)
        for i in range(self.num_conv_layers + 1):
            diff_out = self.diffuse_kernels[i](diff_out)
            #spec_out = self.specular_kernels[i](spec_out)
        return diff_out, spec_out

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

        return tf.reduce_mean(tf.abs(denoised - original))


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
