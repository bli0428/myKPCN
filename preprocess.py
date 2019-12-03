import numpy as np
import cv2

def get_data(filepath):
	"""
	for each image, reads in color buffers, color variances, feature buffers, and feature variances
	returns corresponding pairs of noisy and "ground-truth" images
	"""

	# currently hard coding cornell box image
	x = cv2.imread(filepath + '/noisy_cornell_box')
	y = cv2.imread(filepath + '/cornell_box')

	#cv2.imshow('image1', x)
	#cv2.imshow('image2', y)

	#cv2.waitKey(0)

	return x, y

