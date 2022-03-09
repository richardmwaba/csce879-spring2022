import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_data(dataset, kfold, DATA_DIR, BATCH_SIZE=64, BUFFER_SIZE=10000):
    """Load TensorFlow dataset
    Argument:
        dataset: name of tf dataset e.g., 'imdb_reviews'
        kfold: number of folds for cross-validation (the larger the smaller valid set)
        DATA_DIR: directory for reading/writing dataset
        BATCH_SIZE: number of training batches
        BUFFER_SIZE: for shuffling data
    Returns:
        train, valid (both having kfold splits) and test datasets
    """
    
    val_pct = int(100/kfold)
    
    train_ds = tfds.load(dataset, split=[f'train[:{k}%]+train[{k+val_pct}%:]' for k in range(0, 100, val_pct)],
                         data_dir=DATA_DIR, as_supervised=True)
    val_ds = tfds.load(dataset, split=[f'train[{k}%:{k+val_pct}%]' for k in range(0, 100, val_pct)],
                         data_dir=DATA_DIR, as_supervised=True)
    test_ds = tfds.load(dataset, split='test', data_dir=DATA_DIR, as_supervised=True)
    
    # for each fold, shuffle the data for training and create batches of these (text, label) pairs
    for i in range(kfold):
        train_ds[i] = train_ds[i].shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds[i] = val_ds[i].batch(BATCH_SIZE//kfold).prefetch(tf.data.AUTOTUNE)
        
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds


def plot_performance(history, metrics=['accuracy', 'loss']):
    """Plot performance metrics
    Argument:
        history: training history from Keras's model.fit()
        metrics: performance metrics for trained model
    Returns:
        plt figure
    """
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(history.history[metrics[0]])
    plt.plot(history.history['val_'+metrics[0]], '')
    plt.xlabel("Epochs")
    plt.ylabel(metrics[0])
    plt.legend([metrics[0], 'val_'+metrics[0]])
    plt.ylim(None, 1)
    plt.subplot(1, 2, 2)
    plt.plot(history.history[metrics[1]])
    plt.plot(history.history['val_'+metrics[1]], '')
    plt.xlabel("Epochs")
    plt.ylabel(metrics[1])
    plt.legend([metrics[1], 'val_'+metrics[1]])
    plt.ylim(0, None)
    plt.show()
    
    
def plot_confusion_mat(model, x_valid, y_valid, include_values=True):
    """Draw confusion matrix
    Argument:
        model: trained model
        x_valid: np array of validation data
        y_valid: np array of validation label
        includes_values (bool): whether to include values in confusion matrix
    Returns:
        plt figure
    """
    props = model.predict(x_valid) # # if >= 0.0, it is positive review (label 1) else it is negative (label 0)
    y_pred = np.array([1 if p >= 0 else 0 for p in props])
    cm = confusion_matrix(y_valid, y_pred, normalize='true')
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(10,10))
    disp.plot(include_values=include_values, ax=ax, cmap=plt.cm.Blues)
    plt.show()

    
def flatten(PrefetchDataset):
    """Flatten PrefetchDataset (e.g., outputting datasets of load_data) into numpy arrays
    Argument:
        PrefetchDataset: TensorFlow's PrefetchDataset
    Returns:
        1D arrays of texts and its corresponding labels
    """
    text_arr, label_arr = [], []
    
    PrefetchDataset = list(PrefetchDataset)
    nbatches = len(PrefetchDataset[0][0])
    for i in range(nbatches):
        text_arr.extend(PrefetchDataset[i][0].numpy())
        label_arr.extend(PrefetchDataset[i][1].numpy())
        
    return np.array(text_arr), np.array(label_arr)


def confidence_interval(test_acc, test_size):
    """Calculate confidence interval

    Arguments:
        test_acc (float) -- prediction accuracy
        test_size (int) -- size of labels

    Returns:
        list -- confidence interval
    """
    z = 1.96
    class_error = 1 - test_acc

    confidence = z * math.sqrt((class_error * (1 - class_error)) / test_size)
    ci = [class_error-confidence, class_error+confidence]

    return ci