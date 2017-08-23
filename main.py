import tensorflow as tf
import numpy as np
from dataset.dataset import MnistDataset
import math
import pandas as pd
import tarfile
import gzip
import os.path
import matplotlib.pyplot as plt

n_hidden_1 = 4096 # 1st layer number of features
n_hidden_2 = 4096 # 2nd layer number of features
n_hidden_3 = 1500 # 2nd layer number of features
n_hidden_4 = 1000 # 2nd layer number of features
n_hidden_5 = 500 # 2nd layer number of features

n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

X = tf.placeholder(tf.float32, [None, n_input])
Y_ = tf.placeholder(tf.float32, [None, n_classes])
lr = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)

weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='W1'),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='W2'),
    'w23': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3]), name='W23'),
    'w24': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4]), name='W24'),
    'w25': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5]), name='W25'),

    'w3': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name='W3'),
#    'w4': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='W4')
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
    'b23': tf.Variable(tf.random_normal([n_hidden_3]), name='b23'),
    'b24': tf.Variable(tf.random_normal([n_hidden_4]), name='b24'),
    'b25': tf.Variable(tf.random_normal([n_hidden_5]), name='b25'),

    'b3': tf.Variable(tf.random_normal([n_classes]), name='b3'),
    
#    'b4': tf.Variable(tf.random_normal([n_hidden_2]), name='b4')
}


#Single hidden layer
layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_1 = tf.nn.relu(layer_1)

layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_2 = tf.nn.relu(layer_2)

layer_3 = tf.add(tf.matmul(layer_2, weights['w23']), biases['b23'])
layer_3 = tf.nn.relu(layer_3)
#layer_3 = tf.nn.dropout(layer_3, pkeep)

layer_4 = tf.add(tf.matmul(layer_3, weights['w24']), biases['b24'])
layer_4 = tf.nn.relu(layer_4)

layer_5 = tf.add(tf.matmul(layer_4, weights['w25']), biases['b25'])
layer_5 = tf.nn.relu(layer_5)


layer_6 = tf.nn.dropout(layer_2, pkeep)
Ylogits = tf.add(tf.matmul(layer_6, weights['w3']), biases['b3'])
Y = tf.nn.softmax(Ylogits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
#cross_entropy = cross_entropy + 0.01*tf.nn.l2_loss(weights['w3'])
cross_entropy = tf.reduce_mean(cross_entropy)

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
predictions = tf.argmax(Y, 1)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
#train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

if( not os.path.isfile('test/test.csv')):
    with gzip.open('test/test.csv.gz', 'rb') as f:
        file_content = f.read()

    with open('test/test.csv', 'wb') as f:
        f.write(file_content)

if(not os.path.isfile('test/train.csv')):
    with gzip.open('test/train.csv.gz', 'rb') as f:
        file_content = f.read()

    with open('test/train.csv', 'wb') as f:
        f.write(file_content)

history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}
data = MnistDataset(test_filename = 'test/test.csv', train_filename='test/train.csv', generate_test_set=False)
i = 0
while(True):
    i = i + 1
    #data.train.next_batch
    max_learning_rate = 0.01
    max_learning_rate = 0.001
    min_learning_rate = 0.0001
    decay_speed = 1e-6
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    batch_X, batch_Y = data.train.next_batch(1000)

    a, c = sess.run([accuracy, cross_entropy], {X: batch_X, Y_: batch_Y, pkeep: 1.0})
    prev = str(data.train.epoch) + ": training accuracy:" + str(a) + " training loss: " + str(c)
    print(prev)
    history['acc'].append(a)
    history['loss'].append(c)
    #if(generate_test_set and i % 100 == 0):
    #real_X, real_Y = data.test.get_all()
    #acc, ccc = sess.run([accuracy, cross_entropy], {X: real_X, Y_: real_Y, pkeep: 1.0})

    #print(prev +  ": test accuracy:" + str(acc) + " test loss: " + str(ccc) + "\t LR" + str(learning_rate) )
    #history['val_acc'].append(acc)
    #history['val_loss'].append(ccc)
    learning_rate = 0.5
    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate, pkeep: 0.5})
    if data.train.epoch == 40:
        break

# generate plots
fig, ax = plt.subplots(2,1)
ax[0].plot(history['loss'], color='b', label="Training loss")
ax[0].plot(history['val_loss'], color='r', label='Validation loss', axes = ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history['acc'], color='b', label="Training accuracy")
ax[1].plot(history['val_acc'], color='r', label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.show()
# Run test data
batch_X = data.test.inputs
p = sess.run([predictions], {X: batch_X, pkeep: 1.0})
results = pd.DataFrame({'ImageId': pd.Series(range(1, len(p[0]) + 1)), 'Label': pd.Series(p[0])})

print(p)
results.to_csv('results.csv', index=False)
