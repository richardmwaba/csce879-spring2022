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
    model.add(MaxPooling2D(padding='same'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(padding='same'))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(200, activation='relu', kernel_regularizer=L2()))
    model.add(Dropout(0.5))
    model.add(Dropout(0.5))
    model.add(Dense(nclass, activation='softmax'))

    return model

def cnn_net_3(input_shape, nclass, **kwargs):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(padding='same'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(padding='same'))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(200, activation='relu', kernel_regularizer=L2()))
    model.add(Dropout(0.5))
    model.add(Dense(nclass, activation='softmax'))

    return model

def res_net(input_shape, nclass, **kwargs):
    architec = []
    filt = 32

    conv_1 = Conv2D(filters=filt, kernel_size=(3, 3), activation='relu', input_shape=input_shape)
    architec.append(conv_1)
    bn_1 = BatchNormalization()
    architec.append(bn_1)

    if 'input_x' in kwargs:
        x = kwargs['input_x']
        filt *= 2

        num_blocks_list = [2, 3, 1]

        for i in range(len(num_blocks_list)):
            num_blocks = num_blocks_list[i]
            for j in range(num_blocks):
                layer = ResBlock(filter_num=filt, stride=2)(x=x, is_training=True)
                architec.append(layer)
            filt *= 2
            
    architec.append(MaxPooling2D(pool_size=(4, 4)))
    architec.append(Flatten())
    architec.append(Dense(nclass, activation='softmax'))
    
    model = Sequential(architec)

    return model