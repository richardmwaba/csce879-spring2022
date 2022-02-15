from gc import callbacks
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from keras.callbacks import EarlyStopping
from tqdm import tqdm
from model_CIFAR import *
from util_CIFAR import *



if len(sys.argv) < 2:
    print('Not enough arguments!')
    print('Usage: python {0} [run_name]'.format(sys.argv[0]))
    sys.exit(1)

# Set configurations
run_name = sys.argv[1]
input_dim = [32, 32, 3]
model_name = 'res_net'
epochs = 10
batch_size = 64
loss_function = 'sparse_categorical_crossentropy'
optimizer = tf.keras.optimizers.Adam()
nclass = 100
DATA_DIR = './tensorflow-datasets/'

# Set path to save model and to save performance plots
model_path = './model/{0}'.format(run_name)
if not os.path.isdir(model_path):
    os.makedirs(model_path, exist_ok=True)
result_path = './result/{0}'.format(run_name)
if not os.path.isdir(result_path):
    os.makedirs(result_path, exist_ok=True)

# Load data and normalize (from [0,255] to [0,1])
images_train, labels_train, images_valid, labels_valid = load_data('cifar100', DATA_DIR)
images_train = np.array(images_train).astype('float32') / 255
images_valid = np.array(images_valid).astype('float32') / 255
labels_train = np.array(labels_train)
labels_valid = np.array(labels_valid)

# Model Init
model = eval(model_name)(input_dim, nclass)
model.compile(loss = loss_function, optimizer = optimizer, metrics=['accuracy'])

# Model Training
history = model.fit(
    images_train, labels_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(images_valid, labels_valid),
    callbacks= [EarlyStopping(monitor='val_accuracy', patience=5)]
)

# Save model
model.save_weights(os.path.join(model_path, 'weights.h5'))

# Plot the performance results and save
fig_acc = show_train_history(history, 'accuracy', 'val_accuracy')
plt.savefig(os.path.join(result_path, 'acc_plot.png'))
fig_los = show_train_history(history, 'loss', 'val_loss')
plt.savefig(os.path.join(result_path, 'loss_plot.png'))