import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, BatchNormalization, Lambda 
from tensorflow.keras.optimizers import Adam, SGD, Adagrad
from tensorflow.keras.activations import relu, softmax

## Create convolutional layer
def convolutional_layers(filters: int, kernel: int, strides: int, num: int, mult_filter: bool = False) -> tuple:
    """Create convolutional layers 

    Arguments:
        filters {int} -- number of filters
        kernel {int} -- size of kernel
        strides {int} -- size of strides
        num {int} -- [description]

    Keyword Arguments:
        mult_filter {bool} -- multiply number of filters by 2 for each layer (default: {False})

    Returns:
        tuple -- (list of convolutional layers, current filter size)
    """
    # Track filter
    current_filter = filters
    layers = []

    for i in range(num):
        hd = Conv2D(filters=current_filter, kernel_size=kernel, strides=strides, padding='same', activation=relu)
        layers.append(hd)

        if mult_filter:
            current_filter *= 2

    return (layers, current_filter)

## Create dense layer
def dense_layer(units: int, dense_num: int, activation: str) -> list:
    """ Create dense layers

    Arguments:
        units {int} -- number of units 
        dense_num {int} -- number of layers
        activation {str} -- activation function

    Returns:
        list -- list of layers created
    """
    layers = []
    for i in range(dense_num):
        layers.append(Dense(units=units, activation=activation))

    return layers 

## Create pooling layer
def pooling_layer() -> MaxPooling2D:
    """ Create max pooling layer

    Returns:
        MaxPooling2D -- max pooling layer
    """
    return MaxPooling2D(padding='same')

## Create batch normalization layer
def batch_normalization() -> BatchNormalization:
    """Create batch normalization layer

    Returns:
        BatchNormalization -- batch normalization layer
    """
    return BatchNormalization()

## Create residual block
def residual_block(filters: int, strides: int, resnum: int, initial_x: tf.Tensor) -> list:
    """Create residual block

    Arguments:
        filters {int} -- number of filters
        strides {int} -- size of strides
        resnum {int} -- number of residual blocks to create

    Returns:
        list -- residual blocks as layers
    """
    layers = []
    for i in range(resnum):
        hd = Lambda(lambda X: ResBlock(filter_num=filters, stride=strides))
        hd_call = hd(x=initial_x, training=True)
        layers.append(hd_call)

    return layers

## Flatten input
def flatten() -> Flatten:
    """Flatten input

    Returns:
        Flatten -- flattened input
    """
    return Flatten()

## Load data
def load_data(dataset, DATA_DIR, partition_split=[90,10]):
    train_ds = tfds.load(dataset,
                         split='train[:{0}%]'.format(partition_split[0]),
                         data_dir=DATA_DIR).shuffle(1024)
    valid_ds = tfds.load(dataset,
                         split='train[-{0}%:]'.format(partition_split[1]),
                         data_dir=DATA_DIR)
    
    return data2numpy(train_ds, valid_ds)

## convert data to numpy
def data2numpy(train_ds, valid_ds):
    images_train, labels_train = [], []
    images_valid, labels_valid = [], []
    
    for ins in train_ds:
        labels_train.append(ins['label'].numpy())
        images_train.append(ins['image'].numpy())
    
    for ins in valid_ds:
        labels_valid.append(ins['label'].numpy())
        images_valid.append(ins['image'].numpy())
        
    # lists of images and labels
    return images_train, labels_train, images_valid, labels_valid

## Plot results
def show_train_history(train_history, train, validation):
    plt.figure()
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


## ResNet
class ResBlock(tf.Module):

    def __init__(self, filter_num, stride=1):
        super().__init__()
        self.stride = stride

        # Both self.conv1 and self.down_conv layers downsample the input when stride != 1
        self.bn1 = BatchNormalization()
        self.conv1 = Conv2D(filters=filter_num, kernel_size=(3, 3), strides=stride, padding="same")
        self.bn2 = BatchNormalization()
        self.conv2 = Conv2D(filters=filter_num,  kernel_size=(3, 3), padding="same")

        if self.stride != 1:
            self.down_conv = Conv2D(filters=filter_num, kernel_size=(1, 1), strides=stride,  padding="same")
            self.down_bn = BatchNormalization()

    def __call__(self, x, is_training):
        identity = x
        if self.stride != 1:
            identity = self.down_conv(identity)
            identity = self.down_bn(identity, training=is_training)

        x = self.conv1(x)
        x = tf.nn.relu(x)
        x = self.bn1(x, training=is_training)
        
        
        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = self.bn2(x, training=is_training)

        return x + identity