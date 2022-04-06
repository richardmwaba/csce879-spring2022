import tensorflow as tf
from keras.models import Sequential
from util_MNIST import *
from keras.layers import Dense, Flatten, Input, Dropout

def model1 ():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(64, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, tf.nn.softmax))
    
    return model


def model2 ():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(64, tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, tf.nn.softmax))
    
    return model


def model3 ():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(64, tf.nn.relu))
    model.add(Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, tf.nn.softmax))
    
    return model


def model4 ():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(256, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, tf.nn.softmax))
    
    return model


def model5 ():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(256, tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, tf.nn.softmax))
    
    return model


def model6 ():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(256, tf.nn.relu))
    model.add(Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, tf.nn.softmax))
    
    return model


def model7 ():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(64, tf.nn.relu))
    model.add(tf.keras.layers.Dense(64, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, tf.nn.softmax))
    
    return model


def model8 ():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(64, tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(tf.keras.layers.Dense(64, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, tf.nn.softmax))
    
    return model


def model9 ():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(64, tf.nn.relu))
    model.add(Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, tf.nn.softmax))
    
    return model


def model10 ():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(256, tf.nn.relu))
    model.add(tf.keras.layers.Dense(256, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, tf.nn.softmax))
    
    return model


def model11 ():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(256, tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, tf.nn.softmax))
    
    return model


def model12 ():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(256, tf.nn.relu))
    model.add(Dropout(0.5))
    model.add(tf.keras.layers.Dense(256, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, tf.nn.softmax))
    
    return model