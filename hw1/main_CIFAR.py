from gc import callbacks
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sys

from keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
from model_CIFAR import *
from util_CIFAR import *



if len(sys.argv) < 2:
    print('Not enough arguments!')
    print('Usage: python {0} [configs_path]'.format(sys.argv[0]))
    sys.exit(1)

config_path = sys.argv[1]
config_pathlist = glob.glob(os.path.join(config_path,'*.yml'))
for i in range(len(config_pathlist)):
    config = load_config(config_pathlist[i])
    
    # Load config from yml file
    input_dim = config['dataset']['input_dim']
    partition_split = config['dataset']['data_split']
    nclass = config['dataset']['nclass']
    DATA_DIR = config['dataset']['directory']
    model_name = config['model']['name']
    epoch = config['train']['epochs']
    batch_size = config['train']['batch_size']
    loss_function = config['loss']['name']
    optimizer = config['optimizer']['name']

    run_name = 'run{0}'.format(i+1)

    # Set path to save model and to save performance plots
    model_path = './model/{0}'.format(run_name)
    if not os.path.isdir(model_path):
        os.makedirs(model_path, exist_ok=True)
    result_path = './result/{0}'.format(run_name)
    if not os.path.isdir(result_path):
        os.makedirs(result_path, exist_ok=True)

    # Load data and normalize (from [0,255] to [0,1])
    images_train, labels_train, images_valid, labels_valid = load_data('cifar100', DATA_DIR, partition_split)
    images_train = np.array(images_train).astype('float32') / 255.0
    images_valid = np.array(images_valid).astype('float32') / 255.0
    labels_train = np.array(labels_train)
    labels_valid = np.array(labels_valid)

    # Model Initialization
    model = eval(model_name)(input_dim, nclass, input_x=images_train)
    model.compile(loss = loss_function, optimizer = optimizer, metrics=['accuracy'])

    # only keep the model that has achieved the best val_accuracy so far
    model_checkpoint_callback = ModelCheckpoint(filepath=model_path,
                                                save_weights_only=True,
                                                monitor='val_accuracy',
                                                mode='max',
                                                save_best_only=True)

    # Model Training
    print("----------------------------------")
    print("Train using model '{}'".format(model_name))
    history = model.fit(
        images_train, labels_train,
        batch_size=batch_size,
        epochs=epoch,
        verbose=1,
        validation_data=(images_valid, labels_valid),
        callbacks= [EarlyStopping(monitor='val_accuracy', patience=5),
                   model_checkpoint_callback]
    )
    print("Training done!")

    model.save_weights(os.path.join(model_path, 'weights.h5'))

    # Plot the performance results and save
    fig_acc = show_train_history(history, 'accuracy', 'val_accuracy')
    plt.savefig(os.path.join(result_path, 'acc_plot.png'))
    fig_los = show_train_history(history, 'loss', 'val_loss')
    plt.savefig(os.path.join(result_path, 'loss_plot.png'))