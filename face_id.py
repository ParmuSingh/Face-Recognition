import tensorflow as tf
from pickle import load
import numpy as np
from PIL import Image
from sklearn.utils import shuffle

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


data = load(open("./data.pkl", "rb"))
# print(data)
# print(data.shape)
n_classes = 3
n_examples = len(data)
batch_size = 10

# Version 1 - 50 epochs
VERSION = 1 
TRAIN_MODEL = True
SAVE_MODEL = True
RESTORE_MODEL = False

# placeholders
image = tf.placeholder('float', [None, 256, 256, 3])
label = tf.placeholder('float', [None, 3])

def column(matrix, i):
    return [row[i] for row in matrix]

weights = {
	'hl5': tf.Variable(tf.random_normal([16*16*256, 1024]), name='w_hl5'),
	'ol' : tf.Variable(tf.random_normal([1024, n_classes]), name='w_ol')
}

biases = {
	'hl5': tf.Variable(tf.random_normal([1024]), name='b_hl5'),
	'ol' : tf.Variable(tf.random_normal([n_classes]), name='b_ol')
}

# building the neural network
conv1 = tf.layers.conv2d(inputs=image, filters=32, kernel_size=[5, 5], padding='same',activation=tf.nn.relu, name='conv1')
conv1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[5, 5], padding='same',activation=tf.nn.relu, name='conv2')
conv2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=[5, 5], padding='same',activation=tf.nn.relu, name='conv3')
conv3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[5, 5], padding='same',activation=tf.nn.relu, name='conv4')
conv4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

fc = tf.reshape(conv4, shape=[-1,16*16*256])
fc = tf.nn.sigmoid(tf.matmul(fc, weights['hl5']) + biases['hl5'], name='fc')

output = tf.add(tf.matmul(fc, weights['ol']), biases['ol'], name='output')

# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=label))

train = tf.train.AdamOptimizer().minimize(loss)

epochs = 20

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	if RESTORE_MODEL: # restore neural network to resume training
		saver = tf.train.import_meta_graph("./saved_models/face_id/r"+str(VERSION)+"/faceid.ckpt.meta")
		saver.restore(sess, tf.train.latest_checkpoint("./saved_models/face_id/r"+str(VERSION)+"/"))

	if not TRAIN_MODEL:
		epochs = 0

	
	for epoch in range(epochs): # train neural network
		epoch_loss = 0

		# data = np.random.shuffle(data)
		data = shuffle(data)
		# print(data)
		ptr = 0
		for iter in range(int(n_examples/batch_size)):
			epoch_data = data[ptr : ptr + batch_size]
			ptr += batch_size

			epoch_x = column(epoch_data, 0)
			epoch_y = column(epoch_data, 1)
			
			# the actual training
			_, err = sess.run([train, loss], feed_dict={image: epoch_x, label: epoch_y})
			epoch_loss += err
		print("Epoch:", epoch+1, "- Completed out of:", epochs, "- Loss: ", epoch_loss)

	if SAVE_MODEL: # save neural network
		save_path = saver.save(sess, "./saved_models/face_id/r"+str(VERSION)+"/faceid.ckpt")
		print("Model saved at : ", save_path)
		tf.train.write_graph(graph_or_graph_def=sess.graph_def, logdir="./", name="savegraph.pbtxt")

	# you can manual test here or restore the model later and test.
