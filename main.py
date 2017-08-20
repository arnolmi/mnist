import tensorflow as tf
import numpy as np
from dataset.dataset import MnistDataset
import math
import pandas as pd

n_hidden_1 = 800 # 1st layer number of features
n_hidden_2 = 200 # 2nd layer number of features

n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

X = tf.placeholder(tf.float32, [None, n_input])
Y_ = tf.placeholder(tf.float32, [None, n_classes])
lr = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)

#layer = tf.layers.dense(X, 255, activation=tf.nn.relu)
#layer = tf.nn.dropout(layer, pkeep)

weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='W1'),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='W2'),
    'w3': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name='W3'),
    'w4': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='W4')
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
    'b3': tf.Variable(tf.random_normal([n_classes]), name='b3'),
    'b4': tf.Variable(tf.random_normal([n_hidden_2]), name='b4')
}


layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_1 = tf.nn.relu(layer_1)

layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_2 = tf.nn.relu(layer_2)

#layer_3 = tf.add(tf.matmul(layer_2, weights['w4']), biases['b4'])
#layer_3 = tf.nn.relu(layer_3)
layer_2 = tf.nn.dropout(layer_2, pkeep)
Ylogits = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
Y = tf.nn.softmax(Ylogits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) *100

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
predictions = tf.argmax(Y, 1)

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

data = MnistDataset(test_filename = 'test/test.csv', train_filename='test/train.csv')
for i in range(1,5000):
    # data.train.next_batch    max_learning_rate = 0.003
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    batch_X, batch_Y = data.train.next_batch(1000)
    #batch_X = batch_X.reshape((100, batch_X.shape[1]))

    #batch_Y = batch_Y.reshape((100,batch_Y.shape[1]))
    a, c = sess.run([accuracy, cross_entropy], {X: batch_X, Y_: batch_Y, pkeep: 1.0})
    if a > 0.999:
        break
    print(str(data.train.epoch) + ": training accuracy:" + str(a) + " training loss: " + str(c) )
    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate, pkeep: 0.75})

# Run test data
batch_X = data.test.inputs
p = sess.run([predictions], {X: batch_X, pkeep: 1.0})
results = pd.DataFrame({'ImageId': pd.Series(range(1, len(p[0]) + 1)), 'Label': pd.Series(p[0])})
print(p)
results.to_csv('results.csv', index=False)
