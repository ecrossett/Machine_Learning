import numpy as np
import tensorflow as tf

# program to find root (ie: minimize cost) of y = w^2 -10w + 25

# initialize variabe to zero
w = tf.Variable(0, dtype = tf.float32)
# define cost function
cost = tf.add(tf.add(w**2, tf.multiply(-10.0, w)), 25)
# define learning algorithm (minimize cost function)
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
seesion = tf.Session()
session.run(init)
print(session.run(w))

