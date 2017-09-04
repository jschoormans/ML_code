# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 21:15:00 2017

@author: jasper
"""


from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np



# CREATE MOCK DATA
xlinspace=np.linspace(0,100,num=1e4)
sinus=np.sin(xlinspace)
sinus=5*(sinus+1); 
modsinus=np.sin(xlinspace)+np.sin(20*xlinspace)+0.2*np.sin(3*xlinspace)


plt.plot(sinus)
plt.plot(modsinus); plt.show()


N2=750 #nr of steps used for estimation
N=9000 #NR OF DATAPOINTS 


xinput2=np.zeros((N,N2))
for i in np.arange(N):
    print(modsinus[i:i+2]);
    xinput2[i]=modsinus[i:i+N2]

yinput2=sinus[1:1+N]
yinput2=np.reshape(yinput2,[N,])
yinput2=np.round(yinput2)

print('shape of xinput',np.shape(xinput2))
print('shape of yinput',np.shape(yinput2))
# Parameters
learning_rate = 0.1
num_steps = 100
batch_size = N
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 10 # MNIST data input (img shape: 28*28)
num_classes = 11 # MNIST total classes (0-9 digits)


# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': xinput2}, y=yinput2,
    batch_size=batch_size, num_epochs=None, shuffle=True)

# Define the neural network
def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['images']
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.layers.dense(x, n_hidden_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
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
n_images = 100
# Get images from test set
test_images = xinput2[:n_images]
# Prepare the input data
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': test_images}, shuffle=False)
# Use the model to predict the images class
preds = list(model.predict(input_fn))

# Display
plt.plot(sinus[:n_images])
plt.plot(5*(1+modsinus[:n_images]))
plt.plot(yinput2[:n_images])
plt.plot(preds,'ro')
plt.show()

