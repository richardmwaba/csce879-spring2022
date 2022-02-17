from cmath import inf
from gc import callbacks
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, Adamax, SGD
from tqdm import tqdm
from model_CIFAR import *
from util_CIFAR import *



# Set configurations
run_name = 'output'
input_dim = [32, 32, 3]
model_names = ['simple_cnn', 'cnn_net_2', 'cnn_net_3']
epochs = 100
batch_sizes = [32, 64]
loss_function = 'sparse_categorical_crossentropy'
optimizers = [Adam(), SGD(), Adamax()]
partition_split = [80,20]

nclass = 100
DATA_DIR = './tensorflow-datasets/'

# Set path to save model and to save performance plots
model_path = './model/{0}'.format(run_name)
if not os.path.isdir(model_path):
    os.makedirs(model_path, exist_ok=True)
result_path = './result/{0}'.format(run_name)
if not os.path.isdir(result_path):
    os.makedirs(result_path, exist_ok=True)


# Model Initialization and training
trained_models = {}
training_histories = {}


# Load data and normalize (from [0,255] to [0,1])
images_train, labels_train, images_valid, labels_valid = load_data('cifar100', DATA_DIR, partition_split)
images_train = np.array(images_train).astype('float32') / 255
images_valid = np.array(images_valid).astype('float32') / 255
labels_train = np.array(labels_train)
labels_valid = np.array(labels_valid)


for model_name in model_names:
    for optimizer in optimizers:
        for batch_size in batch_sizes:
            model = eval(model_name)(input_dim, nclass, input_x=images_train)
            model.compile(loss = loss_function, optimizer = optimizer, metrics=['accuracy'])

            # Model Training
            print("----------------------------------")
            print("Train using model '{}'".format(model_name))
            history = model.fit(
                images_train, labels_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(images_valid, labels_valid),
                callbacks= [EarlyStopping(monitor='val_accuracy', patience=5)]
            )
            print("Training done!")

        # Add last accuracy to final accuracies dictionary    
        final_acc = history.history['val_accuracy'][-1]
        
        # Add model to trained models dictionary
        model_desc= "{}_{}_{}".format(model_name, optimizer, batch_size)
        trained_models[final_acc] = [model, model_desc]
        # Add history to trained histories dictionary
        training_histories[final_acc] = history

# Get model with maximum validation accuracy
best_model = trained_models[max(trained_models)]


# Save best model
best_model[0].save_weights(os.path.join(model_path, "{}_weights.h5".fomat(best_model[1])))

# Plot confusion matrix
confusion_matrix_fig = show_confusion_mat(model=best_model[0], x_valid=images_valid, y_valid=labels_valid,class_names=range(nclass))
plt.savefig(os.path.join(result_path, 'confusion_matrix.png'))

# Plot the performance results and save
fig_acc = show_train_history(training_histories[max(trained_models)], 'accuracy', 'val_accuracy')
plt.savefig(os.path.join(result_path, 'acc_plot.png'))
fig_los = show_train_history(history, 'loss', 'val_loss')
plt.savefig(os.path.join(result_path, 'loss_plot.png'))

