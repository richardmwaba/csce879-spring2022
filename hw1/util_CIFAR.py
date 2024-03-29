import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_data(dataset, DATA_DIR, partition_split=[90,10]):
    """_summary_
    Arguments:
        dataset {_type_} -- _description_
        DATA_DIR {_type_} -- _description_
    Keyword Arguments:
        partition_split {list} -- _description_ (default: {[90,10]})
    Returns:
        _type_ -- _description_
    """
    train_ds = tfds.load(dataset,
                         split='train[:{0}%]'.format(partition_split[0]),
                         data_dir=DATA_DIR).shuffle(1024)
    valid_ds = tfds.load(dataset,
                         split='train[-{0}%:]'.format(partition_split[1]),
                         data_dir=DATA_DIR)

    test_ds = tfds.load(dataset,
                         split='test',
                         data_dir=DATA_DIR)
                         
    
    return data2numpy(train_ds, valid_ds, test_ds)


def data2numpy(train_ds, valid_ds, test_ds):
    """_summary_
    Arguments:
        train_ds {_type_} -- _description_
        valid_ds {_type_} -- _description_
    Returns:
        _type_ -- _description_
    """
    images_train, clabels_train, flabels_train = [], [], []
    images_valid, clabels_valid, flabels_valid = [], [], []
    images_test, clabels_test, flabels_test = [], [], []
    
    for ins in train_ds:
        clabels_train.append(ins['coarse_label'].numpy())
        flabels_train.append(ins['label'].numpy())
        images_train.append(ins['image'].numpy())
    
    for ins in valid_ds:
        clabels_valid.append(ins['coarse_label'].numpy())
        flabels_valid.append(ins['label'].numpy())
        images_valid.append(ins['image'].numpy())

    for ins in test_ds:
        clabels_test.append(ins['coarse_label'].numpy())
        flabels_test.append(ins['label'].numpy())
        images_test.append(ins['image'].numpy())
        
    # lists of images and labels
    return images_train, clabels_train, flabels_train, images_valid, clabels_valid, flabels_valid, images_test, clabels_test, flabels_test


def show_train_history(train_history, train, validation):
    """_summary_
    Arguments:
        train_history {_type_} -- _description_
        train {_type_} -- _description_
        validation {_type_} -- _description_
    """
    plt.figure()
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    

def show_confusion_mat(model, x_valid, y_valid, include_values=False):
    """Draw confusion matrix
    Argument:
        model: trained model
        x_valid: np array of validation data
        y_valid: np array of validation label
        includes_values (bool): whether to include values in confusion matrix
    Returns:
        plt figure
    """
    props = model.predict(x_valid)
    y_pred = np.argmax(props,axis=1)  # convert hot vectors to predicted labels in [0, 99]
    cm = confusion_matrix(y_valid, y_pred, normalize='true')
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(10,10))
    disp.plot(include_values=include_values, ax=ax, cmap=plt.cm.Blues)
    plt.show()

def confidence_interval(test_acc, labels_test):
    z = 1.96
    class_error = 1 - test_acc

    confidence = z * math.sqrt((class_error * (1 - class_error)) / labels_test.shape[0])
    ci = [class_error-confidence, class_error+confidence]

    return ci