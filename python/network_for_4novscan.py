# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 14:36:15 2017

@author: jaspe
"""


from __future__ import print_function
import numpy as np
import os 
import tensorflow as tf
import matplotlib.pyplot as plt

#%%
class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    Note: it would be more interesting to use a HyperOpt search space:
    https://github.com/hyperopt/hyperopt
    """

    def __init__(self, X_train, X_test):
        # Input data
        self.train_count = len(X_train)  # 7352 training series
        self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = len(X_train[0])  # 128 time_steps per series

        # Training
        self.learning_rate = 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 300
        self.batch_size = 1000

        # LSTM structure
        self.n_inputs = len(X_train[0][0])  # Features count is of 9: 3 * 3D sensors features over time
        self.n_hidden = 32  # nb of neurons inside the neural network
        self.n_classes = 5  # Final output classes
        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random_normal([self.n_classes]))
        }
#%%
def LSTM_Network(_X, config):
    """Function returns a TensorFlow RNN with two stacked LSTM cells
    Two LSTM cells are stacked which adds deepness to the neural network.
    Note, some code of this notebook is inspired from an slightly different
    RNN architecture used on another dataset, some of the credits goes to
    "aymericdamien".
    Args:
        _X:     ndarray feature matrix, shape: [batch_size, time_steps, n_inputs]
        config: Config for the neural network.
    Returns:
        This is a description of what is returned.
    Raises:
        KeyError: Raises an exception.
      Args:
        feature_mat: ndarray fature matrix, shape=[batch_size,time_steps,n_inputs]
        config: class containing config of network
      return:
              : matrix  output shape [batch_size,n_classes]
    """
    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, config.n_inputs])
    # new shape: (n_steps*batch_size, n_input)

    # Linear activation
    _X = tf.nn.relu(tf.matmul(_X, config.W['hidden']) + config.biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, config.n_steps, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    #J: ADDED RESUE=TRUE, BECAUSE 2nd time running network gave error 
    
    with tf.variable_scope('lstm1'): #J: ADDED THIS< DONT KNOW WHY / WHAT IS SCOPE
        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    
    with tf.variable_scope('lstm2'): #J: ADDED THIS< DONT KNOW WHY / WHAT IS SCOPE
        lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    
    with tf.variable_scope('lstm3'): #J: ADDED THIS< DONT KNOW WHY / WHAT IS SCOPE
        lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output']


def one_hot(y_):
    """
    Function to encode output labels from number indexes.
    E.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    """
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) +1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


#%%---------------------------
#load data
#---------------------------


path='C:\\Users\\jaspe\\Dropbox\\phD\\python\\4novscan_6'
os.chdir(path)
labeldata=np.load('answer.npy')
kspace=np.load('data.npy')


for i in np.arange(np.size(kspace,axis=1)):
    temp=kspace[:,i]
    temp=temp-np.min(temp)
    temp=temp/np.max(temp)
    kspace[:,i]=temp

print('labeldata shape: ',np.shape(labeldata))
print('kspace shape: ',np.shape(kspace))

#%%---------------------------
#make training tensors
#---------------------------

N=1000;
Ntest=300;
N2=10; #length of datasets
N3=30

X_train=np.zeros([N,N2,N3])
X_test=np.zeros([Ntest,N2,N3])

for i in np.arange(N):
    X_train[i,:,:]=kspace[i:i+N2,:N3]

for i in np.arange(Ntest):
    X_test[i,:,:]=kspace[N+i:N+i+N2,:N3]

print('shape of X_train:',np.shape(X_train))
print('shape of X_test:',np.shape(X_test))

y_train=labeldata[:,:N]
y_test=labeldata[:,N:N+Ntest]
print('shape of y_train:',np.shape(y_train))
print('shape of y_test:',np.shape(y_test))


y_train=np.reshape(y_train,[N,1])
print('shape of y_train:',np.shape(y_train)) 
y_test=np.reshape(y_test,[Ntest,1])
print('shape of y_test:',np.shape(y_test)) 

y_train=one_hot(y_train)
print('shape of y_train:',np.shape(y_train)) #one class too many??
y_test=one_hot(y_test)
print('shape of y_test:',np.shape(y_test)) #one class too many??



config = Config(X_train, X_test)

# test normalization (to do: fix it!)
print("Some useful info to get an insight on dataset's shape and normalisation:")
print("features shape, labels shape, each features mean, each features standard deviation")
print(X_train.shape, y_train.shape,
          np.mean(X_train), np.std(X_train))
print('minimum val, maximum val:',np.min(X_train),np.max(X_train))
print("the dataset is therefore properly normalised, as expected.")
    

#%%---------------------------
# building the network...
#---------------------------
print('Building network...') 

X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
Y = tf.placeholder(tf.float32, [None, config.n_classes])

pred_Y = LSTM_Network(X, config)

# Loss,optimizer,evaluation
l2 = config.lambda_loss_amount * \
        sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
# Softmax loss and L2
cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred_Y)) + l2
optimizer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
#%%---------------------------
# training the network...
#---------------------------

# Note that log_device_placement can be turned ON but will cause console spam with RNNs.
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
init = tf.global_variables_initializer()
sess.run(init)

best_accuracy = 0.0

print('Starting training...')
# Start training for each batch and loop epochs
for i in range(config.training_epochs):
    for start, end in zip(range(0, config.train_count, config.batch_size),
                          range(config.batch_size, config.train_count + 1, config.batch_size)):
        sess.run(optimizer, feed_dict={X: X_train[start:end],
                                       Y: y_train[start:end]})

    # Test completely at every epoch: calculate accuracy
    pred_out, accuracy_out, loss_out = sess.run(
        [pred_Y, accuracy, cost],
        feed_dict={
            X: X_test,
            Y: y_test
        }
    )
    print("traing iter: {},".format(i) +
          " test accuracy : {},".format(accuracy_out) +
          " loss : {}".format(loss_out))
    best_accuracy = max(best_accuracy, accuracy_out)

print("")
print("final test accuracy: {}".format(accuracy_out))
print("best epoch's test accuracy: {}".format(best_accuracy))
print("")

#%%

plt.plot(pred_out.argmax(1),'.')
plt.plot(y_test.argmax(1),'r-')
plt.show()

plt.plot(pred_out.argmax(1),y_test.argmax(1),'b*')
plt.show()



