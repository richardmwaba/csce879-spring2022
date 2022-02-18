import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
DATA_DIR = './tensorflow-datasets/'

def load_data(dataset):

    train_ds = tfds.load(dataset, split='train[:90%]', data_dir=DATA_DIR).shuffle(1024)
    validation_ds = tfds.load(dataset, split='train[-10%:]', data_dir=DATA_DIR)
    test_ds = tfds.load(dataset, split='test', data_dir=DATA_DIR)
    
    images_train, labels_train = [], []
    images_valid, labels_valid = [], []
    images_test, labels_test = [], []
    for ins in train_ds:
        labels_train.append(ins['label'].numpy())
        images_train.append(ins['image'].numpy())

    for ins in validation_ds:
        labels_valid.append(ins['label'].numpy())
        images_valid.append(ins['image'].numpy())

    for ins in test_ds:
        labels_test.append(ins['label'].numpy())
        images_test.append(ins['image'].numpy())
        
    images_train = np.array(images_train).astype('float32') / 255
    images_valid = np.array(images_valid).astype('float32') / 255
    labels_train = np.array(labels_train)
    labels_valid = np.array(labels_valid)
    images_test = np.array(images_test)
    labels_test = np.array(labels_test)
        
    return images_train, labels_train, images_valid, labels_valid, images_test, labels_test




def show_train_history(train, validation):

    plt.figure()
    plt.plot(train)
    plt.plot(validation)
    plt.title('Train History')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
