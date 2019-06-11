import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EPOCHS = 5000
BATCH_SIZE = 5
LEARNING_RATE = 0.005
# Creation of data:

x = np.linspace(-1, 1, num=500)
y = x + np.random.uniform(-1, 1, 500) * np.random.randint(1,10)

# Creation of placeholders and Variables

x_in = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_in = tf.placeholder(shape=[None, 1], dtype=tf.float32)
W = tf.Variable(tf.random_normal([1,1]))
b = tf.Variable(tf.random_normal([1,1]))

# output of the model (prediction)

model_output = tf.add(tf.matmul(x_in, W), b)

# Optimization

loss = tf.reduce_mean(tf.square(y_in - model_output))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

# training time!
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_vec = []
    for epoch in range(EPOCHS):
        random_i = np.random.choice(len(x), size=BATCH_SIZE)
        x_data = np.transpose([x[random_i]])
        y_data = np.transpose([y[random_i]])
        
        sess.run(train_step, feed_dict={x_in: x_data, y_in: y_data})
        epoch_loss = sess.run(loss, feed_dict={x_in: x_data, y_in: y_data})
        loss_vec.append(epoch_loss)
        if (epoch) % 10 == 0:
            print("Step {} out of {} Loss {}".format(epoch, EPOCHS, epoch_loss))

    [slope] = sess.run(W)
    [y_intercept] = sess.run(b) 

# Getting the parameters of best fit line

best_fit = [slope*i+y_intercept for i in x]

# Plotting time!

plt.plot(x, y, 'o', label='Data points')
plt.plot(x, best_fit, 'r-', label='BEST_FIT_LINE')
plt.show()
# plot of loss fluctuations
plt.plot(np.arange(EPOCHS), loss_vec, 'b')
plt.show()