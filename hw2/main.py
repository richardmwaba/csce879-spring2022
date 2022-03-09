from __future__ import print_function

import numpy as np
import time
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
run_name = 'run_test'
model_names = ['lstm_rnn', 'gru_rnn']
dense_units = [64, 128]
kfold = 5
epoch = 10
batch_size = 64
lr = 1e-4
DATA_DIR = './tensorflow-datasets/'

# Set path to save performance plots
result_path = './result/{0}'.format(run_name)
if not os.path.isdir(result_path):
    os.makedirs(result_path, exist_ok=True)

# Load data
train_ds, valid_ds, test_ds = load_data('imdb_reviews', kfold, DATA_DIR, batch_size)  # 25,000 for train+valid, 25,000 for testing

# Model Initialization and training
trained_models = {}
training_histories = {}
max_accuracy = float(-np.inf)

for model_name in model_names:
    for units in dense_units:
        k_val_acc = []  # all valid acc across k folds
        k_train_acc = [] # all training accuracies across k folds

        tic = time.perf_counter() # get current time
        # k-fold cross-validation
        for i in range(kfold):
            print("Starting fold {} for model '{}'".format(i+1, model_name))

            # model init
            model = eval(model_name)(train_ds[i], dense_units=units)

            model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                        optimizer=Adam(lr),
                        metrics=['accuracy'])

            # model training
            history = model.fit(train_ds[i],
                                epochs=epoch,
                                validation_data=valid_ds[i],
                                validation_steps=5)  # When an epoch ends, validation generator will yield validation_steps batches

            print('Fold {} finished!'.format(i+1))

            # Save fold metrics
            fold_train_acc = history.history['accuracy'][-1]
            fold_val_acc = history.history['val_accuracy'][-1]

            k_val_acc.append(fold_val_acc)  # record final valid acc of the fold
            k_train_acc.append(fold_train_acc)

        toc = time.perf_counter() # get current time
        # Record training time
        train_time = f'{toc-tic:.4f} secs'

        # Get average metrics
        train_acc = np.mean(k_train_acc)
        val_acc = np.mean(k_val_acc)

        # Add model to trained models dictionary
        model_desc= f"{model_name}-units_{units}"
        trained_models[model_desc] = [model, train_acc, val_acc, k_val_acc, k_train_acc, train_time]

        # # Add history to trained histories dictionary
        # training_histories[model_desc] = [k_train_acc, k_val_acc, k_train_loss, k_val_loss]

        # Add best model
        if val_acc > max_accuracy:
            max_accuracy = val_acc
            trained_models['best_model'] = [model, train_acc, val_acc, model_desc]

# Get model with maximum validation accuracy
best_model = trained_models['best_model'][0]
best_model_desc = trained_models['best_model'][3]

# Write all trained models to file
trained_models_fh = os.path.join(result_path, 'trained_models.txt')
with open(trained_models_fh, 'w') as outfile:
    for key, val in trained_models.items():
        if key == 'best_model': 
            continue
        outfile.write('*************************************************************************************************\n')
        outfile.write(f'{key}\n')
        outfile.write('---------------------------------------------------------------------------------------\n')
        outfile.write(f"> Training accuracy {val[1]:.4f} Validation accuracy: {val[2]:.4f} Training time: {val[-1]}\n")
        outfile.write('*************************************************************************************************\n')

# Write models and their kfold results to file
trained_models_folds_fh = os.path.join(result_path, 'trained_models_folds.txt')
with open(trained_models_folds_fh, 'w') as outfile:
    for key, val in trained_models.items():
        if key == 'best_model': 
            continue
        outfile.write('******************************************************************************************\n')
        outfile.write(f'{key} - Score per fold \n')
        for i in range(len(val[3])):
            outfile.write('--------------------------------------------------------------------------------\n')
            outfile.write(f'> Fold {i+1} - Train accuracy: {val[4][i]:.4f} - Validation accuracy: {val[3][i]:.4f}\n')
        outfile.write('******************************************************************************************\n')

# Evaluate model on test data
final_score = best_model.evaluate(test_ds)

# flatten test_ds into numpy text array and label array
test_texts, test_labels = flatten(test_ds)  

# Calculate confidence interval
test_acc = final_score[1]   
confidence_int = confidence_interval(test_acc=test_acc, test_size=test_labels.shape[0])

# Write to output
with open(trained_models_fh, 'a') as outfile:
    outfile.write("\n----------------------------------------------------------------------------------------------------------------\n")
    outfile.write(f"The final test accuracy is {test_acc:.4f} with 95% confidence interval of {confidence_int}")


# # # Plot performance results and confusion matrix and save
# fig = plot_performance(training_histories[best_model_desc])
# plt.savefig(os.path.join(result_path, 'performance_plot.png'))

cm_fig = plot_confusion_mat(best_model, test_texts, test_labels)
plt.savefig(os.path.join(result_path, 'cm_plot.png'))