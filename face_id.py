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
TRAIN_MODEL = False
SAVE_MODEL = True
RESTORE_MODEL = True

# placeholders
image = tf.placeholder('float', [None, 256, 256, 3])
label = tf.placeholder('float', [None, 3])

def column(matrix, i):
    return [row[i] for row in matrix]

# def conv2d(x, W):
# 	# filter dimension = [filter_height, filter_width, in_channels, out_channels]
# 	# strides = [1, stride, stride, 1]
# 	# The stride of the sliding window for each dimension of input.
# 	# Filter size is known from W
# 	# strides determines how much the window shifts by in each of the dimensions. 
# 	return tf.layers.conv2d(x, W, strides=[1,1,1,1], padding='SAME') # tf.nn.conv2d is lower level and doen't have activation parameter

# def maxpool2d(x):
# 	return tf.layers.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

weights = {
	# 'hl1': tf.Variable(tf.random_normal([5, 5, 3, 32]), name='w_hl1'),
	# 'hl2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name='w_hl2'),
	# 'hl3': tf.Variable(tf.random_normal([5, 5, 64, 128]), name='w_hl3'),
	# 'hl4': tf.Variable(tf.random_normal([5, 5, 128, 256]), name='w_hl4'),
	'hl5': tf.Variable(tf.random_normal([16*16*256, 1024]), name='w_hl5'),
	'ol' : tf.Variable(tf.random_normal([1024, n_classes]), name='w_ol')
}

biases = {
	# 'hl1': tf.Variable(tf.random_normal([32]), name='b_hl1'),
	# 'hl2': tf.Variable(tf.random_normal([64]), name='b_hl2'),
	# 'hl3': tf.Variable(tf.random_normal([128]), name='b_hl3'),
	# 'hl4': tf.Variable(tf.random_normal([256]), name='b_hl4'),
	'hl5': tf.Variable(tf.random_normal([1024]), name='b_hl5'),
	'ol' : tf.Variable(tf.random_normal([n_classes]), name='b_ol')
}

# conv1 = tf.add(conv2d(image, weights['hl1']), biases['hl1'], name='conv1')
# conv1 = maxpool2d(conv1)

# conv2 = tf.add(conv2d(conv1, weights['hl2']), biases['hl2'], name='conv2')
# conv2 = maxpool2d(conv2)

# conv3 = tf.add(conv2d(conv2, weights['hl3']), biases['hl3'], name='conv3')
# conv3 = maxpool2d(conv3)

# conv4 = tf.add(conv2d(conv3, weights['hl4']), biases['hl4'], name='conv4')
# conv4 = maxpool2d(conv4)

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


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=label))

train = tf.train.AdamOptimizer().minimize(loss)

epochs = 20

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	if RESTORE_MODEL:
		saver = tf.train.import_meta_graph("E:/workspace_py/saved_models/face_id/r"+str(VERSION)+"/faceid.ckpt.meta")
		saver.restore(sess, tf.train.latest_checkpoint("E:/workspace_py/saved_models/face_id/r"+str(VERSION)+"/"))

	if not TRAIN_MODEL:
		epochs = 0

	
	for epoch in range(epochs):
		epoch_loss = 0

		# data = np.random.shuffle(data)
		data = shuffle(data)
		# print(data)
		ptr = 0
		for iter in range(int(n_examples/batch_size)):
			epoch_data = data[ptr : ptr + batch_size]
			ptr += batch_size
			# epoch_x = epoch_data[:, 0]
			# epoch_y = epoch_data[:, 1]
			epoch_x = column(epoch_data, 0)
			epoch_y = column(epoch_data, 1)
			# print("shapes:")
			# print(len(epoch_y))
			# np.set_printoptions(threshold=np.nan)
			# print(epoch_x)
			# print(epoch_y)
			# for i in range(10):
			# print(epoch_x[i].shape)
			# print(epoch_y.shape)
			_, err = sess.run([train, loss], feed_dict={image: epoch_x, label: epoch_y})
			epoch_loss += err
		print("Epoch:", epoch+1, "- Completed out of:", epochs, "- Loss: ", epoch_loss)

	if SAVE_MODEL:
		save_path = saver.save(sess, "E:/workspace_py/saved_models/face_id/r"+str(VERSION)+"/faceid.ckpt")
		print("Model saved at : ", save_path)
		tf.train.write_graph(graph_or_graph_def=sess.graph_def, logdir="./", name="savegraph.pbtxt")

	img = Image.open(open("E:/Datasets!!/family_pics/papa/papa122.jpg", "rb"))
	img = np.asarray(img)
	img = img/255.0

	print("test:")
	prediction = sess.run(output, feed_dict={image: [img]})
	print(prediction)
	prediction = np.argmax(prediction)
	if prediction == 0:
		prediction = "papa"
	elif prediction == 1:
		prediction = "mummy"
	else:
		prediction = "prince"

	print("Prediction:",prediction)

	img = Image.open(open("E:/Datasets!!/family_pics/mummy/mummy134.jpg", "rb"))
	img = np.asarray(img)
	img = img/255.0

	print("test:")
	prediction = sess.run(output, feed_dict={image: [img]})
	print(prediction)
	prediction = np.argmax(prediction)
	if prediction == 0:
		prediction = "papa"
	elif prediction == 1:
		prediction = "mummy"
	else:
		prediction = "prince"

	print("Prediction:",prediction)

	img = Image.open(open("E:/Datasets!!/family_pics/prince/prince27.jpg", "rb"))
	img = np.asarray(img)
	img = img/255.0

	print("test:")
	prediction = sess.run(output, feed_dict={image: [img]})
	print(prediction)
	prediction = np.argmax(prediction)
	if prediction == 0:
		prediction = "papa"
	elif prediction == 1:
		prediction = "mummy"
	else:
		prediction = "prince"

	print("Prediction:",prediction)
