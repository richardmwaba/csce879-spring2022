import tensorflow as tf
from keras.models import Sequential
from util_MNIST import *
from keras.layers import Dense


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, tf.nn.relu))
model.add(tf.keras.layers.Dense(
        128, tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.L1(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01)))
model.add(tf.keras.layers.Dense(
        256, tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.L1(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01)))
model.add(tf.keras.layers.Dense(10, tf.nn.softmax))
optimizer = tf.keras.optimizers.Adam()



