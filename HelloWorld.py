# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 00:05:40 2017
@author: beecoder

"""

import tensorflow as tf

# Declare a constant string Hello World. This appears as an op in the Graph
hello = tf.constant("Hello World")

#Define sess as a Tensorflow Session
sess = tf.Session()

# Tensorflow graph structures are first defined and then run by using the eval
# or run functions
print (sess.run(hello))



