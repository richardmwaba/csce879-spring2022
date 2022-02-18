import tensorflow as tf
from keras.models import Sequential
from util_MNIST import *
from keras.layers import Dense, Flatten, Input

def model1 ():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(64, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, tf.nn.softmax))
    
    return model


def model3 ():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(256, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, tf.nn.softmax))
    
    return model


def model5 ():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(64, tf.nn.relu))
    model.add(tf.keras.layers.Dense(64, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, tf.nn.softmax))
    
    return model


def model7 ():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(256, tf.nn.relu))
    model.add(tf.keras.layers.Dense(256, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, tf.nn.softmax))
    
    return model
