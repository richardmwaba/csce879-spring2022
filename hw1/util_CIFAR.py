import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import ruamel.yaml
import hashlib
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
    
    return data2numpy(train_ds, valid_ds)


def data2numpy(train_ds, valid_ds):
    """_summary_
    Arguments:
        train_ds {_type_} -- _description_
        valid_ds {_type_} -- _description_
    Returns:
        _type_ -- _description_
    """
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
    

def show_confusion_mat(model, x_valid, y_valid, class_names):
    """Draw confusion matrix
    Argument:
        model: trained model
        x_valid: np array of validation data
        y_valid: np array of validation label
        class_names: range(nclass) array
    Returns:
        plt figure
    """
    props = model.predict(x_valid)
    y_pred = np.argmax(props,axis=1)  # convert hot vectors to predicted labels in [0, 99]
    cm = confusion_matrix(y_valid, y_pred, normalize='true')
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

######################################################################
class ResBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super().__init__()
        self.stride = stride

        # Both self.conv1 and self.down_conv layers downsample the input when stride != 1
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            padding="same")

        if self.stride != 1:
            self.down_conv = tf.keras.layers.Conv2D(filters=filter_num,
                                                    kernel_size=(1, 1),
                                                    strides=stride,
                                                    padding="same")
            self.down_bn = tf.keras.layers.BatchNormalization()

    def __call__(self, x, is_training):
        identity = x
        if self.stride != 1:
            identity = self.down_conv(identity)
            identity = self.down_bn(identity, training=is_training)

        x = self.bn1(x, training=is_training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        
        
        x = self.bn2(x, training=is_training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        return x + identity