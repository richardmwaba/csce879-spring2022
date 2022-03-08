from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

from tensorflow.keras.optimizers import Adam, Adamax, SGD
from scipy import stats
from model import *
from util import *


# Set configurations
run_name = 'run_toy'
model_names = ['lstm_rnn', 'gru_rnn']
kfold = 10
epoch = 1
batch_size = 64
lr = 1e-4
DATA_DIR = './tensorflow-datasets/'

# Set path to save performance plots
result_path = './result/{0}'.format(run_name)
if not os.path.isdir(result_path):
    os.makedirs(result_path, exist_ok=True)

# Load data
train_ds, valid_ds, test_ds = load_data('imdb_reviews', kfold, DATA_DIR, batch_size)  # 25,000 for train+valid, 25,000 for testing

avg_val_acc = []  # all avg valid acc across the models/architectures
for model_name in model_names:
    k_val_acc = []  # all valid acc across k folds
    # k-fold cross-validation
    for i in range(kfold):
        print("Starting fold {} for model '{}'".format(i+1, model_name))

        # model init
        model = eval(model_name)(train_ds[i])
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=Adam(lr),
                      metrics=['accuracy'])

        # model training
        history = model.fit(train_ds[i],
                            epochs=epoch,
                            validation_data=valid_ds[i],
                            validation_steps=5)  # When an epoch ends, validation generator will yield validation_steps batches

        print('Fold {} finished!'.format(i+1))
        k_val_acc.append(history.history['val_accuracy'][-1])  # record final valid acc of the fold
    
    print('Average valid accuracy across {} folds: {}'.format(kfold, np.mean(k_val_acc)))
    avg_val_acc.append(np.mean(k_val_acc))
    

## extract the model/architecture that returns the highest value in avg_val_acc and test it on test_ds and plot etc.


# # Plot performance results and confusion matrix and save
# fig = plot_performance(training_histories[best_model_desc])
# plt.savefig(os.path.join(result_path, 'performance_plot.png'))

# test_texts, test_labels = flatten(test_ds)  # flatten test_ds into numpy text array and label array
# cm_fig = plot_confusion_mat(best_model, test_texts, test_labels)
# plt.savefig(os.path.join(result_path, 'cm_plot.png'))
