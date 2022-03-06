from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

from tensorflow.keras.optimizers import Adam, Adamax, SGD
from sklearn.model_selection import KFold
from scipy import stats
from model import *
from util import *


# Set configurations
run_name = 'run_toy'
model_names = ['simple_rnn']
epoch = 1
batch_size = 64
lr = 1e-4
num_folds = 5
DATA_DIR = './tensorflow-datasets/'

# Set path to save performance plots
result_path = './result/{0}'.format(run_name)
if not os.path.isdir(result_path):
    os.makedirs(result_path, exist_ok=True)

# Load data
train_ds, test_ds = load_data('imdb_reviews', DATA_DIR, batch_size)  # 25,000 for training, 25,000 for testing

# Define Kfold 
kfold = KFold(n_splits=num_folds, shuffle=True)

# Model initialization and training
trained_models = {}
training_histories = {}
max_accuracy = -np.inf

fold_num = 1
for train, val in kfold.split(train_ds):
    for model_name in model_names:
        # We can include hyperparameters here
        model = eval(model_name)(train_ds[train])
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    optimizer=Adam(lr),
                    metrics=['accuracy'])

        # Model training
        print(f'Training using model {model_name} for fold {fold_num}'.center(50, '*'))
        history = model.fit(train_ds[train],
                            epochs=epoch,
                            validation_data=train_ds[val],
                            validation_steps=30)  # When an epoch ends, validation generator will yield validation_steps batches
        # Save fold metrics
        fold_train_acc = history.history['accuracy'][-1]
        fold_val_acc = history.history['val_accuracy'][-1]

        # Condifence interval per fold
        confidence_int = confidence_interval(test_acc = history.history['val_accuracy'][-1], test_size=train_ds[val].shape[0])

        # Add model to trained models dictionary
        model_desc= f"{fold_num}-{model_name}"
        trained_models[model_desc] = [model, fold_train_acc, fold_val_acc, confidence_int]

        # Add best model
        if fold_val_acc > max_accuracy:
            max_accuracy = fold_val_acc
            trained_models['best_model'] = [model, fold_train_acc, fold_val_acc, confidence_int, model_desc]

        print("Training done!")
    
    fold_num += 1

# Get model with maximum validation accuracy
best_model = trained_models['best_model'][0]
best_model_desc = trained_models['best_model'][4]


# Write all trained models to file
trained_models_fh = os.path.join(result_path, 'trained_models.txt')
with open(trained_models_fh, 'w') as outfile:
    for key, val in trained_models.items():
        outfile.write(f"{key}\t\t\t{val[1]}\t\t\t{val[2]}\t\t\t{val[3]}\n")

# Evaluate model on test data
final_score = best_model.evaluate(test_ds)

# Calculate confidence interval
test_acc = final_score[1]
confidence_int = confidence_interval(test_acc=test_acc, test_size=test_ds.shape[0])

# Write to output
with open(trained_models_fh, 'a') as outfile:
    outfile.write("\n----------------------------------------------------------------------------------------------------------------\n")
    outfile.write("The final test accuracy is {} with 95% confidence interval of {}".format(test_acc, confidence_int))

# Plot performance results and confusion matrix and save
fig = plot_performance(training_histories[best_model_desc])
plt.savefig(os.path.join(result_path, 'performance_plot.png'))

test_texts, test_labels = flatten(test_ds)  # flatten test_ds into numpy text array and label array
cm_fig = plot_confusion_mat(best_model, test_texts, test_labels)
plt.savefig(os.path.join(result_path, 'cm_plot.png'))