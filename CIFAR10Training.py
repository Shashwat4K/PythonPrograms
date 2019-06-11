import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import _pickle as cPickle
import time
print('Program strated at {}'.format(time.time()))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)
STEPS = 1000
BATCH_SIZE = 50
DATA_PATH = '/home/shashwat/Documents/CIFAR10-Dataset/'

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def conv2d(x, W, name):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

def max_pool_2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def conv_layer(ip, shape, name):
    W = weight_variable(shape, name+'_Weights')
    b = bias_variable([shape[3]], name+'_biases')
    return tf.nn.relu(conv2d(ip, W, name) + b)

def full_layer(ip, size, name):
    in_size = int(ip.get_shape()[1])
    W = weight_variable([in_size, size], name+'_Weights')
    b = bias_variable([size], name+'_biases')
    return tf.add(tf.matmul(ip, W), b, name=name)

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
print('Data loaded at {}'.format(time.time()))
print('Number of train images: {}'.format(len(cifar.train.images)))
print('Number of train labels: {}'.format(len(cifar.train.labels)))
print('Number of test images: {}'.format(len(cifar.test.images)))
print('Number of test labels: {}'.format(len(cifar.test.labels)))

images = cifar.train.images
# display_cifar(images, 10)

datastart = time.time()
print('Data placeholders and Variables creation start at {}'.format(datastart))

# input image
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='train_images') # 32(width)x32(height)x3(channels (rgb))

# expected output
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='target_output')
keep_prob = tf.placeholder(tf.float32)

# Layer 1 input: 32*32*3 image 
conv1 = conv_layer(x, shape=[5,5,3,32], name='Conv1')
conv1_pool = max_pool_2x2(conv1, name='Conv1Pool')
# Layer 1 output: 16*16*32

# Layer 2 input: 16*16*32 image
conv2 = conv_layer(conv1_pool, shape=[5,5,32,64], name='Conv2')
conv2_pool = max_pool_2x2(conv2, name='Conv2Pool')
# Layer 2 output: 8*8*64
# Flat layer input: 8*8*64 
conv2_flat = tf.reshape(conv2_pool, [-1, 8*8*64], name='Conv2_Flat')

# Applying rectified Linear Unit
full1 = tf.nn.relu(full_layer(conv2_flat, 1024, name='full1'), name='full1_ReLU')
# Drop some neurons on the basis of keep probability
full1_drop = tf.nn.dropout(full1, keep_prob = keep_prob, name='full1_Dropout')

full2 = tf.nn.relu(full_layer(full1_drop, 512, name='full2'), name='full2_ReLU')
full2_drop = tf.nn.dropout(full1, keep_prob = keep_prob, name='full2_Dropout')

full3 = tf.nn.relu(full_layer(full2_drop, 256, name='full3'), name='full3_ReLU')
full3_drop = tf.nn.dropout(full3, keep_prob = keep_prob, name='full3_Dropout')

full4 = tf.nn.relu(full_layer(full3_drop, 128, name='full4'), name='full4_ReLU')
full4_drop = tf.nn.dropout(full4, keep_prob=keep_prob, name='full4_Dropout')
# Here we have the neurons required to construct final full layer
y_conv = full_layer(full4_drop, 10, name='Predicted_output')
# full layer constructed 'y_conv'

# Calculate the loss using cross_entropy function 
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv, labels=y_))
# Use AdamOpimizer to minimize the cross_entropy loss
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
# Calculate how many no. of times correct predictions were made by the model
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# saver = tf.train.Saver()

def test(sess):
    X = cifar.test.images.reshape(10, 1000, 32, 32, 3)
    Y = cifar.test.labels.reshape(10, 1000, 10)
    acc = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], keep_prob: 1.0}) for i in range(10)])	# Changing the keep probabiltity to 0.8 for testing the accuracy change
    print('Mean Accuracy: {:4}%'.format(acc*100)) 
train_start = time.time()
print('Training begins...')
print('Started Training at {}'.format(train_start))
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
    sess.run(tf.global_variables_initializer())
    loss_vec = []
    for i in range(STEPS):
        epoch_x, epoch_y = cifar.train.next_batch(BATCH_SIZE)
        _ , temp_loss = sess.run([train_step, cross_entropy], feed_dict={x:epoch_x, y_:epoch_y, keep_prob:1.0})
        loss_vec.append(temp_loss)
        print('Step {} out of {}'.format(i+1, STEPS))
        '''
        if (i+1)%100 == 0:
            saver.save(sess, save_path='./CIFAR10Saver/epoch_'+str(i+1)+'_variables.ckpt')
            print('Saved!', str(i))
        '''  
    train_end = time.time()
    print('Trianing concluded in {}'.format(train_end - train_start))
    print('Now Testing...')
    print('Testing starts at {}'.format(time.time()))      
    test(sess)
    print('testing ends at {}'.format(time.time()))

print('Minimium loss occured at Step# {}'.format( np.argmin(np.array(loss_vec)) ))

plt.plot(list(np.arange(0, STEPS)), loss_vec, 'r')
plt.xlabel('Step#')
plt.ylabel('Loss')
plt.title('Loss vs Step graph')
plt.show()