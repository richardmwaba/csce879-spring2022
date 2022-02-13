import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt


def load_data(dataset, DATA_DIR, partition_split=[90,10]):
    train_ds = tfds.load(dataset,
                         split='train[:{0}%]'.format(partition_split[0]),
                         data_dir=DATA_DIR).shuffle(1024)
    valid_ds = tfds.load(dataset,
                         split='train[-{0}%:]'.format(partition_split[1]),
                         data_dir=DATA_DIR)
    
    return data2numpy(train_ds, valid_ds)


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


def show_train_history(train_history, train, validation):
    plt.figure()
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()