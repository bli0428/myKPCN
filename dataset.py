import numpy as np
import cv2


def read_image(filepath):
	pass

def read_exr(filepath):
	pass

def write_image(filepath):
	pass

def write_exr(filepath):
	pass

def make_dataset():
	pass

def calulate_variance(f):
	pass

def calculate_gradient(f):
	pass

def preprocess_diffuse(c):
	pass

def preprocess_specular(c):
	pass

def get_data(filepath):
	"""
	for each image, reads in diffuse and specular color buffers, color variances, feature buffers, and feature variances
	returns corresponding pairs of noisy and "ground-truth" images
	"""

	# currently hard coding cornell box image
	x = cv2.imread(filepath + '/noisy_cornell_box') # 100 samples (will end up being 32 or 128)
	y = cv2.imread(filepath + '/cornell_box')       # 1000 samples (will end up being 1024)

	#cv2.imshow('image1', x)
	#cv2.imshow('image2', y)

	#cv2.waitKey(0)

	'''
	will eventually return diffuse matrix of MxNx(3+D), 3 for RGB and  D for the number of additional feature buffers,
			       specular matrix of MxNx(3+D), and reference image (MxNx3)
	'''

	return x, y
	# return diffuse, specular, y
