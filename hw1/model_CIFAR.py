import tensorflow as tf
from keras.models import Sequential
from util_CIFAR import *
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, Dense, BatchNormalization, Lambda
from keras.regularizers import L1, L2, L1L2


def simple_cnn(input_shape, nclass, **kwargs):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(nclass, activation='softmax'))

    return model

def cnn_net_2(input_shape, nclass, **kwargs):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(padding='same'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(padding='same'))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(nclass, activation='softmax'))

    return model

def cnn_net_3(input_shape, nclass, **kwargs):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(padding='same'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(padding='same'))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(padding='same', strides=2))
    model.add(Flatten())
    model.add(Dense(200, activation='relu', kernel_regularizer=L2()))
    model.add(Dropout(0.5))
    model.add(Dense(nclass, activation='softmax'))

    return model

def cnn_net_4(input_shape, nclass, **kwargs):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(padding='same', strides=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nclass, activation='softmax'))

    return model