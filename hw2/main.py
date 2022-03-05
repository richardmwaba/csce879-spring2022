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
model_name = 'simple_rnn'
epoch = 1
batch_size = 64
lr = 1e-4
DATA_DIR = './tensorflow-datasets/'

# Set path to save performance plots
result_path = './result/{0}'.format(run_name)
if not os.path.isdir(result_path):
    os.makedirs(result_path, exist_ok=True)

# Load data
train_ds, test_ds = load_data('imdb_reviews', DATA_DIR, batch_size)  # 25,000 for training, 25,000 for testing

# model init
model = eval(model_name)(train_ds)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=Adam(lr),
              metrics=['accuracy'])

# model training
history = model.fit(train_ds,
                    epochs=epoch,
                    validation_data=test_ds,
                    validation_steps=30)  # When an epoch ends, validation generator will yield validation_steps batches

# Plot performance results and confusion matrix and save
fig = plot_performance(history)
plt.savefig(os.path.join(result_path, 'performance_plot.png'))

test_texts, test_labels = flatten(test_ds)  # flatten test_ds into numpy text array and label array
cm_fig = plot_confusion_mat(model, test_texts, test_labels)
plt.savefig(os.path.join(result_path, 'cm_plot.png'))

