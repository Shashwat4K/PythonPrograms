import os
import tensorflow as tf
import tensorboard
import matplotlib.pyplot as plt
import _pickle as cPickle
import numpy as np
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Program Begins now!")
EPOCHS = 1000
BATCH_SIZE = 50
DATA_PATH = '/home/shashwat/Documents/CIFAR10-Dataset/'
LOG_DIR = '/home/shashwat/Documents/NFT/CIFAR10/Logs'

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)

def unpickle(_file):
    with open(os.path.join(DATA_PATH, _file), 'rb') as fo:
        di = cPickle.load(fo, encoding='latin1') # ascii was causing bug here so changed to latin1 encoding
    return di

def one_hot(vec, vals=10):
    out = np.zeros((len(vec), vals))
    out[range(len(vec)), vec] = 1
    return out

def display_cifar(images, size):
        n = len(images)
        plt.figure()
        plt.gca().set_axis_off()
        im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)]) for i in range(size)])
        plt.imshow(im)
        plt.show()

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def nn_layer(input_t, size, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            input_size = int(input_t.get_shape()[1])
            weights = weight_variable([input_size, size])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([size])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.add(tf.matmul(input_t, weights), biases)
            tf.summary.histogram('pre-activations', preactivate)
        if act!=None:
            activations = act(preactivate, name='activation')
        else:
            activations = preactivate    
        tf.summary.histogram('activations', activations)
        return activations                        

def cnn_layer(input_t, shape, layer_name, act=tf.nn.relu, strides=[1,1,1,1], padding='SAME'):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights_cnn'):
            weights = weight_variable(shape)
            variable_summaries(weights)
        with tf.name_scope('biases_cnn'):
            biases = bias_variable([shape[3]])
            variable_summaries(biases)
        with tf.name_scope('Convolution'):
            preactivate = tf.nn.conv2d(input_t, weights, strides=strides, padding=padding) + biases
            tf.summary.histogram('pre-activate', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations    

def pooling_layer(input_t, layer_name, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'):
    with tf.name_scope(layer_name):
        maxpool = tf.nn.max_pool(input_t, ksize, strides, padding)
        return maxpool

def feed_dict_maker(cifar):
    xs, ys = cifar.train.next_batch(BATCH_SIZE)
    k = 1.0
    return {x: xs, y: ys, keep_prob: k}        

# Class to load CIFAR 10 dataset from the directory
class CifarLoader(object):
    def __init__(self, source_files):
        self._source = source_files
        self._index = 0
        self.images = None
        self.labels = None

    def load(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack([d['data'] for d in data])
        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).transpose(0,2,3,1)\
            .astype(float) / 255
        self.labels = one_hot(np.hstack([d['labels'] for d in data]), 10)
        return self

    def next_batch(self, batch_size):
        x, y = self.images[self._index : self._index + batch_size], self.labels[self._index : self._index + batch_size]
        self._index = (self._index + batch_size) % len(self.images)
        return x, y

class CifarDataManager(object):
    def __init__(self):
        self.train = CifarLoader(['data_batch_{}'.format(i) for i in range(1, 6)]).load()
        self.test = CifarLoader(['test_batch']).load()

cifar = CifarDataManager()

trainX = cifar.train.images
trainY = cifar.train.labels
testX = cifar.test.images
testY = cifar.test.labels
print("Data Loaded!")

with tf.name_scope('training_placeholders'):
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)
print("Placeholders created!")

conv_layer1 = cnn_layer(x, shape=[5,5,3,32], layer_name='Conv1')
conv_pool1 = pooling_layer(conv_layer1, layer_name='Conv1_Pool')

conv_layer2 = cnn_layer(conv_pool1, shape=[5,5,32,64], layer_name='Conv2')
conv_pool2 = pooling_layer(conv_layer2, layer_name='Conv2_Pool')

conv2_flat = tf.reshape(conv_pool2, [-1, 8*8*64], name='Conv2_Flat')

full1 = nn_layer(conv2_flat, 1024, 'full1_ReLU')
full_drop1 = tf.nn.dropout(full1, keep_prob = keep_prob, name='full1_dropout')

full2 = nn_layer(full_drop1, 512, 'full2_ReLU')
full_drop2 = tf.nn.dropout(full2, keep_prob=keep_prob, name='full2_dropout')

full3 = nn_layer(full_drop2, 256, 'full3_ReLU')
full_drop3 = tf.nn.dropout(full3, keep_prob=keep_prob, name='full3_dropout')

full4 = nn_layer(full_drop3, 128, 'full4_ReLU')
full_drop4 = tf.nn.dropout(full4, keep_prob=keep_prob, name='full4_dropout')

y_conv = nn_layer(full_drop4, 10, layer_name='Predicted_output', act=None)
print("Layers created!")

with tf.name_scope('cross_entropy'):
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv, labels=y))
tf.summary.scalar('cross_entropy', cross_entropy)
print("Loss function defined!")

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
print("Optimizer defined!")

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(LOG_DIR + '/train', tf.get_default_graph())
test_writer = tf.summary.FileWriter(LOG_DIR + '/test', tf.get_default_graph())
print("Log writers created!")

init = tf.global_variables_initializer()

def test(sess):
    print("Starting Testing procedure...")
    X = cifar.test.images.reshape(10, 1000, 32, 32, 3)
    Y = cifar.test.labels.reshape(10, 1000, 10)
    acc = np.mean([sess.run(accuracy, feed_dict={x: X[i], y: Y[i], keep_prob: 1.0}) for i in range(10)]) # Changing the keep probabiltity to 0.8 for testing the accuracy change
    summary = sess.run(merged, feed_dict={x: cifar.test.images, y: cifar.test.labels, keep_prob: 1.0})
    print('Mean Accuracy: {:4}%'.format(acc*100)) 
    test_writer.add_summary(summary)
    print("Testing completed!")

with tf.Session() as sess:
    sess.run(init)
    print("Training Variables initialized!")
    print("Training begins now...")
    for epoch in range(EPOCHS):
        print("Starting epoch {} out of {}".format(epoch, EPOCHS))
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict_maker(cifar))
        train_writer.add_summary(summary, epoch)
    print("Testing begins now...")
    # test(sess)
