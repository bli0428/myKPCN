from dataset import get_data
from model import Denoise

import numpy as np
import tensorflow as tf
import argparse
from imageio import imwrite

# will change hyperparameters later
parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=100, help='Number of epochs to run [default: 10]')
parser.add_argument('--batch_size', type=int, default=5, help='Batch size [default: 5]')
parser.add_argument('--patch_size', type=int, default=100, help='Patch size [default: 100]')
parser.add_argument('--learning_rate', type=int, default=0.0001, help='Learning rate [default: e-5]')
parser.add_argument('--epsilon', type=int, default=0.00316, help='Epsilon [default: 0.00316]')
parser.add_argument('--log_dir', type=str, default='log', help='Log dir [default: log]')
parser.add_argument('--dataset', type=str, default='images', help='Dataset path  [default: images]')
FLAGS = parser.parse_args()

NUM_EPOCH = FLAGS.num_epoch
BATCH_SIZE = FLAGS.batch_size
LEARNING_RATE = FLAGS.learning_rate
EPSILON = FLAGS.epsilon
LOG_DIR = FLAGS.log_dir
DATASET = FLAGS.dataset

def train():
	'''
	general structure for training
	'''
	inputs, labels = get_data(DATASET)
	model = Denoise()
	optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
	#saver = tf.train.Saver()

    
	s = 'images/original_cornell_box.png'
	imwrite(s, inputs)
    
	inputs = tf.reshape(inputs, (1, 512, 512, 3))/255
	labels = tf.reshape(labels, (1, 512, 512, 3))/255
	for epoch in range(NUM_EPOCH):
        for i in range(0, len(inputs) - BATCH_SIZE, BATCH_SIZE):
            for j in range(0, inputs.shape[1] - PATCH_SIZE, PATCH_SIZE):
                for k in range(0, inputs.shape[2] - PATCH_SIZE, PATCH_SIZE):
                    with tf.GradientTape() as tape:
                        batch_patch_inputs = inputs[i:i+BATCH_SIZE][j:j+PATCH_SIZE][k:k+PATCH_SIZE]
                        batch_patch_labels = labels[i:i+BATCH_SIZE][j:j+PATCH_SIZE][k:k+PATCH_SIZE]
                        diffuse, specular = model.call(batch_patch_inputs, batch_patch_labels)
                        predictions = EPSILON * diffuse + tf.exp(specular) - 1
                        loss = model.loss(predictions, batch_patch_labels)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    print("LOSS", epoch, ":", loss)
                    

	# dont quite remember how to save, need sessions?
	# save_path = saver.save(, os.path.join(LOG_DIR, "model.ckpt"))
	# print("Model saved in file: %s" % save_path)
	write_prediction(inputs, model)

def write_prediction(inputs, model):
	prediction_diff, prediction_spec = model.call(inputs, inputs)
	prediction = np.array(EPSILON * prediction_diff * 255) # Change when we add specular
	prediction = prediction.astype(np.uint8)
	prediction = np.clip(np.reshape(prediction, (512, 512, 3)), 0, 255)

	s = 'images/predicted_cornell_box.png'
	imwrite(s, prediction)


if __name__ == '__main__':
	train()