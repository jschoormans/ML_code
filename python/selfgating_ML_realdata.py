# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 17:05:12 2017

@author: jschoormans
"""
from __future__ import print_function
import numpy as np
import os 
import tensorflow as tf
import matplotlib.pyplot as plt



path='L:\\basic\divi\Projects\cosart\CS_simulations\MachineLearning\\4novscan_6'
os.chdir(path)
respdata=np.load('answer.npy')
ctrksp=np.load('data.npy')


ctrksp=ctrksp/np.max(ctrksp)

plt.plot(ctrksp[:,1])
plt.show()

print(ctrksp.shape)
print(respdata.shape)


num_classes = 4# MNIST total classes (0-9 digits)

# CREATE MOCK DATA
respdata_label=respdata-1
respdata_label=np.reshape(respdata_label,[1360])
plt.plot(respdata_label)
plt.show()


N2=128 #nr of steps used for estimation
N=1200 #NR OF DATAPOINTS 
nchans=1;

ctrkspshape=np.shape(ctrksp)
xinput2=np.zeros([N,N2])
for i in np.arange(N):
    xinput2[i,:]=ctrksp[i:i+N2,2]

yinput2=respdata_label[1:1+N]

print(xinput2.shape)

plt.plot(respdata_label[1:5*N2])
plt.plot(xinput2[1,:])
plt.show()


print('shape of xinput',np.shape(xinput2))
print('shape of yinput',np.shape(yinput2))
# Parameters
learning_rate = 0.15
num_steps = 2500
batch_size = N
display_step = 10

# Network Parameters
n_hidden_1 = 128 # 1st layer number of neurons
n_hidden_2 = 64 # 2nd layer number of neurons
n_hidden_3 = 64 # 3rd layer number of neurons
n_hidden_4 = 64 # 3rd layer number of neurons
n_hidden_5 = 64 # 3rd layer number of neurons

num_input = 10 # MNIST data input (img shape: 28*28)


# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': xinput2}, y=yinput2,
    batch_size=batch_size, num_epochs=None, shuffle=True)

# Define the neural network
def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['images']
    # Hidden fully connected layer with 32 neurons
    layer_1 = tf.layers.dense(x, n_hidden_1)
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    layer_3 = tf.layers.dense(layer_2, n_hidden_3)
    layer_4 = tf.layers.dense(layer_3, n_hidden_4)
    layer_5 = tf.layers.dense(layer_4, n_hidden_5)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer

# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    
    # Build the neural network
    logits = neural_net(features)
    
    # Predictions
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)
    
    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes) 
        
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
    
    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=yinput2, predictions=pred_classes)
    
    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
                  mode=mode,
      predictions=pred_classes,
      loss=loss_op,
      train_op=train_op,
      eval_metric_ops={'accuracy': acc_op})

    return estim_specs


# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Train the Model
model.train(input_fn, steps=num_steps)


# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': xinput2}, y=yinput2,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
model.evaluate(input_fn)




# Predict single images
n_images = 500
# Get images from test set
test_images = xinput2[:n_images]
# Prepare the input data
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': test_images}, shuffle=False)
# Use the model to predict the images class
preds = list(model.predict(input_fn))

# Display
plt.plot(yinput2[:n_images])
plt.plot(preds,'ro')
plt.show()

plt.plot(preds,yinput2[:n_images],'b.')
plt.show()