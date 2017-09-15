# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 09:43:12 2017

@author: beecoder
"""
import tensorflow as tf
#Basic operations in Tensorflow 
#tensorflow has variables and constants  

#start a sessions  
with tf.Session() as sess:
     a = tf.constant(2)
     b = tf.constant(3)
     #a and b are constants, lets use run() mehos to evaluate them 
     print("a is 2 and b is 3")
     print("A + B = %d" % sess.run(a+b))
     print("A * B = %d" % sess.run(a*b))
     

X = tf.placeholder(tf.int16)
Y = tf.placeholder(tf.int16)

#Define some operations
add = tf.add(X, Y) 
multiply = tf.multiply(X,Y) 

#If X and Y arent initialized we will get an error. 
# Placeholders need to be initialized using the feed_dict property.
with tf.Session() as sess:
    print("The addition of x and y is %d" % sess.run(add, feed_dict={X:2,Y:3})) 
    print("The multiplication of x and y is %d" % sess.run(multiply, feed_dict={X:2,Y:3}))

#Matrix multiplication: 
# pay attention to the way dimensions are created in matrices    
m1 = tf.constant([[2., 3.]])  #-. 1 x 2 
m2 = tf.constant([[3.],[2.]]) #-> 2 x 1

#define matric multiplication
product = tf.matmul(m1, m2)

with tf.Session() as sess:
    result = sess.run(product)
    print(result)
    


    
    
