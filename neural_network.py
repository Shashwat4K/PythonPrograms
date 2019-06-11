import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/home/shashwat/Documents/NFT/mnist", one_hot=True)

nodes_hl1 = 500
nodes_hl2 = 500
nodes_hl3 = 500

output_classes = 10

batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')
#y = tf.Variable('float')
def neural_network_model(data):

	h1_layer = {
		'weights': tf.Variable(tf.random_normal([784, nodes_hl1])),
		'biases': tf.Variable(tf.random_normal([nodes_hl1]))
	}

	h2_layer = {
		'weights': tf.Variable(tf.random_normal([nodes_hl1, nodes_hl2])),
		'biases': tf.Variable(tf.random_normal([nodes_hl2]))
	}

	h3_layer = {
		'weights': tf.Variable(tf.random_normal([nodes_hl2, nodes_hl3])),
		'biases': tf.Variable(tf.random_normal([nodes_hl3]))
	}

	output_layer = {
		'weights': tf.Variable(tf.random_normal([nodes_hl3, output_classes])),
		'biases': tf.Variable(tf.random_normal([output_classes]))
	}
	# (input * weights) + biases

	l1 = tf.add(tf.matmul(data, h1_layer['weights']), h1_layer['biases'])
	l1 = tf.nn.relu(l1)
	l2 = tf.add(tf.matmul(l1, h2_layer['weights']), h2_layer['biases'])
	l2 = tf.nn.relu(l2)
	l3 = tf.add(tf.matmul(l2, h3_layer['weights']), h3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
	return output

def train(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	num_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(num_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples / batch_size)):
				e_x, e_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x: e_x, y: e_y})
				epoch_loss += c

			print("Epoch", epoch, "out of", num_epochs, "done! Loss:", epoch_loss)

			correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

			accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print("Accuracy :", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


tf.logging.set_verbosity(old_v)

train(x)	

# tf.logging.set_verbosity(old_v)			
