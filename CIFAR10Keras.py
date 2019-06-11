import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import _pickle as cPickle
import tensorflow.keras as keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
DATA_PATH = '/home/shashwat/Documents/CIFAR10-Dataset/'

def unpickle(_file):
    with open(os.path.join(DATA_PATH, _file), 'rb') as fo:
        di = cPickle.load(fo, encoding='latin1') # ascii was causing bug here so changed to latin1 encoding
    return di

def one_hot(vec, vals=10):
    out = np.zeros((len(vec), vals))
    out[range(len(vec)), vec] = 1
    return out

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
        self.images = images.reshape(n, 3, 32, 32).transpose(0,2,3,1).astype(float) / 255
        self.labels = one_hot(np.hstack([d['labels'] for d in data]), 10)
        return self

    def next_batch(self, batch_size):
        x, y = self.images[self._index : self._index + batch_size], self.labels[self._index : self._index + batch_size]
        self._index = (self._index + batch_size) % len(self.images)
        return x, y

class CifarDataManager(object):
    def __init__(self):
        a  = CifarLoader(['data_batch_{}'.format(i) for i in range(1, 6)])
        b  = CifarLoader(['test_batch'])
        self.train = a.load()
        self.test = b.load()

cifar = CifarDataManager()

trainX = cifar.train.images
trainY = cifar.train.labels
testX  = cifar.test.images
testY = cifar.test.labels

print('Length trainX->{}, testX->{} trainY->{} testY->{}'.format(len(trainX), len(testX), len(trainY), len(testY)))


my_model = keras.Sequential()
my_model.add(keras.layers.Conv2D(filters=96, input_shape=cifar.train.images[0, :, :, :].shape, kernel_size=(3,3)))
my_model.add(keras.layers.Activation('relu'))
my_model.add(keras.layers.Conv2D(filters=96, strides=2, kernel_size=(3,3)))
my_model.add(keras.layers.Activation('relu'))
my_model.add(keras.layers.Dropout(0.2))
my_model.add(keras.layers.Conv2D(filters=192, kernel_size=(3,3)))
my_model.add(keras.layers.Activation('relu'))
my_model.add(keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=2))
my_model.add(keras.layers.Activation('relu'))
my_model.add(keras.layers.Dropout(0.5))
my_model.add(keras.layers.Flatten())
my_model.add(keras.layers.BatchNormalization())
my_model.add(keras.layers.Dense(256))
my_model.add(keras.layers.Activation('relu'))
my_model.add(keras.layers.Dense(10, activation='softmax'))

my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
EPOCHS = 25
BATCH_SIZE = 256

H = my_model.fit(x=trainX, y=trainY, validation_data = (testX, testY), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=None) 
