import math
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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


def show_acc_history(train, validation):

    plt.figure()
    plt.plot(train)
    plt.plot(validation)
    plt.title('Train History')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    
def show_loss_history(train, validation):

    plt.figure()
    plt.plot(train)
    plt.plot(validation)
    plt.title('Train History')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    
def c_m(images_test, model, labels_test):
    
    props = model.predict(images_test)
    y_pred = np.argmax(props,axis=1)
    a = confusion_matrix(labels_test, y_pred)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=a)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
    
    
def confidence_interval(test_acc):

    z = 1.96
    error = 1 - test_acc
    cf = z * math.sqrt ((error*test_acc)/10000)
    ci1 = error - cf
    ci2 = error + cf
    ci = [ci1,ci2]
    return ci
