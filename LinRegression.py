# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 12:27:53 2017

@author: beecoder
"""

# Starting a new project - Implementing Linear regression

#Import libraries  
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

#Define parameters for the model. 
training_rate = 0.01 
learning_epochs = 1000
display_rate = 50 

#STAGE 1 : Static graph building 
#Define the Input/outputs for the model. 
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
num_samples = train_X.shape[0]

# Start defining the model. placeholders, variables, predicted values and 
# finally the cost function

X = tf.placeholder('float')  #Just the type is defined, not the size, size will
                             # become important while assigning values to X
Y = tf.placeholder('float') 

W = tf.Variable(np.random.randn(), name='Weight')
b = tf.Variable(np.random.randn(), name='bias')

pred = tf.add(tf.multiply(X,W), b)
cost = tf.reduce_sum(tf.pow((Y - pred),2)) / (2*num_samples)

optimizer = tf.train.GradientDescentOptimizer(training_rate).minimize(cost)
init = tf.global_variables_initializer()

#STAGE 2 : DYnamic graph building
with tf.Session() as sess: 
    sess.run(init) #start initializing all the variables
    for epoch in range(learning_epochs):   # For each laerning epoch/iteration
        for (x,y) in zip(train_X, train_Y): #for each sample in trng set
            sess.run(optimizer, feed_dict={X:x, Y:y})
        # calculate cost every 50 steps
        if (epoch+1) % 50 == 0 :
            c = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
            print("Epoch: ", "%04d" %(epoch+1), "{:.9f}".format(c), "W = %f" % \
                 sess.run(W), "b = %f" % sess.run(b))
    print("training completed!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    pred_Y = train_X * sess.run(W) + sess.run(b) 
    #Plot the graph
    plt.plot(train_X, train_Y, 'ro',  label="original data")
    plt.plot(train_X, pred_Y, label="fitted_data" )
    plt.legend()
    plt.show()






















'''















# Learning -. Variable objects are trainable=True by default.
# Include Libraries: 
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

# Define parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50 

#Define input values ( not yes placeholders) 
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
num_samples = train_X[0]

#Define input parameters for your model
X = tf.placeholder('float')    #scalar, unlike X_train
Y = tf.placeholder('float')

#Define network model parameters. THese will be trainable by the network
W = tf.Variable(np.random.randn, name='Weight')
b = tf.Variable(np.random.randn, name='bias')

#Define the predicted value of the model
pred = tf.add(tf.multiply(X, W), b)

#Define the cost function to be minimized
cost = tf.reduce_sum(tf.pow((pred - Y), 2))/(2*num_samples) # ???? 

#Define optimizer 
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

#Start training 

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
'''





