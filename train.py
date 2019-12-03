from preprocess import get_data
from model import Model

import numpy as np
import tensorflow as tf
import argparse

# will change hyperparameters later
parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=10 help='Number of epochs to run [default: 10]')
parser.add_argument('--batch_size', type=int, default=5, help='Number of epochs to run [default: 32]')
parser.add_argument('--learning_rate', type=int, default=0.0001, help='Learning rate [default: e-5]')
parser.add_argument('--log_dir', type=str, default='log', help='Log dir [default: log]')
parser.add_argument('--dataset', type=str, default='images', help='Dataset path  [default: images]')
FLAGS = parser.parse_args()

NUM_EPOCH = FLAGS.num_epoch
BATCH_SIZE = FLAGS.batch_size
LEARNING_RATE = FLAGS.learning_rate
LOG_DIR = FLAGS.log_dir
DATASET = FLAGS.dataset

def train():
	'''
	'''
	inputs, labels = get_data(DATASET)
	model = Model()
	optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
	saver = tf.train.Saver()

	for 
		for i in range(0, len(inputs) - BATCH_SIZE, BATCH_SIZE):
			batch_inputs = inputs[i:i+BATCH_SIZE]
			batch_labels = labels[i:i+BATCH_SIZE]
			with tf.GradientTape() as tape:
				predictions = model(batch_inputs)
				loss = model.loss(predictions, batch_labels)
			gradients = tape.gradient(loss, model.trainable_variables)
			model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
	
	save_path = saver.save(, os.path.join(

if __name__ == '__main__':
	train()


