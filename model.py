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
        pass
        
    def call(self, images):
        pass

    def loss(self, denoised, original):
        """
        Computes the loss for a batch of images.

        :param denoised: a 4d matrix (batchsize, width, height, 3) of pixel values representing a batch of image, the output of call
        :param original: a 4d matrix (batchsize, width, height, 3) of pixel values representing a batch of image, the input of call
        :return: loss, a TensorFlow scalar
        """
        denoised = tf.dtypes.cast(denoised, tf.float32)
        original = tf.dtypes.cast(original, tf.float32)
        
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
        