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
from scipy import stats
from model_CIFAR import *
from util_CIFAR import *



# Set configurations
run_name = 'cifar_100_output'
input_dim = [32, 32, 3]
model_names = ['simple_cnn', 'cnn_net_2', 'cnn_net_3', 'cnn_net_4']
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
max_accuracy = float(-inf)


# Load data and normalize (from [0,255] to [0,1])
images_train, clabels_train, flabels_train, images_valid, clabels_valid, flabels_valid, images_test, clabels_test, flabels_test = load_data('cifar100', DATA_DIR, partition_split)
images_train = np.array(images_train).astype('float32') / 255
images_valid = np.array(images_valid).astype('float32') / 255
images_test = np.array(images_test).astype('float32') / 255
clabels_train, flabels_train = np.array(clabels_train), np.array(flabels_train)
clabels_valid, flabels_valid = np.array(clabels_valid), np.array(flabels_valid)
clabels_test, flabels_test = np.array(clabels_test), np.array(flabels_test)


for model_name in model_names:
    for optimizer in optimizers:
        for batch_size in batch_sizes:
            model = eval(model_name)(input_dim, nclass, input_x=images_train)
            model.compile(loss = loss_function, optimizer = optimizer, metrics=['accuracy'])

            # Model Training
            print("----------------------------------")
            print("Train using model '{}'".format(model_name))
            history = model.fit(
                images_train, flabels_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(images_valid, flabels_valid),
                callbacks= [EarlyStopping(monitor='val_accuracy', patience=5)]
            )

            # Add last accuracy to final accuracies dictionary    
            final_acc = history.history['val_accuracy'][-1]
            train_acc = history.history['accuracy'][-1]

            # Confindence interval
            confidence_int = confidence_interval(test_acc=final_acc, labels_test=flabels_valid)

            # Add model to trained models dictionary
            model_desc= "{}_{}_{}".format(model_name, optimizer, batch_size)
            trained_models[model_desc] = [model, train_acc, final_acc, confidence_int]
            # Add best model
            if final_acc > max_accuracy:
                max_accuracy = final_acc
                trained_models['best_model'] = [model, train_acc, final_acc, confidence_int, model_desc]
            # Add history to trained histories dictionary
            training_histories[model_desc] = history

            print("Training done!")

# Get model with maximum validation accuracy
best_model = trained_models['best_model'][0]
best_model_desc = trained_models['best_model'][4]

# Write or trained models to file
trained_models_fh = os.path.join(result_path, 'trained_models.txt')
with open(trained_models_fh, 'w') as outfile:
    for key, val in trained_models.items():
        outfile.write("{}\t\t\t{}\t\t\t{}\t\t\t{}\n".format(key, val[1], val[2], val[3]))


# Save best model
best_model.save_weights(os.path.join(model_path, "{}_weights.h5".format(best_model_desc)))

# Evaluate model on test data
final_score = best_model.evaluate(images_test, flabels_test)

# Calculate confidence interval
test_acc = final_score[1]
confidence_int = confidence_interval(test_acc=test_acc, labels_test=flabels_test)

print("------------------------------------------------------------------\n")
print("The 95% confidence interval is {}".format(confidence_int))

# Write to output
with open(trained_models_fh, 'a') as outfile:
    outfile.write("\n----------------------------------------------------------------------------------------------------------------\n")
    outfile.write("The final test accuracy is {} with 95% confidence interval of {}".format(test_acc, confidence_int))


# Plot confusion matrix
confusion_matrix_fig = show_confusion_mat(model=best_model, x_valid=images_test, y_valid=flabels_test, include_values=False)
plt.savefig(os.path.join(result_path, 'confusion_matrix.png'))

confusion_matrix_fig_2 = show_confusion_mat(model=best_model, x_valid=images_test, y_valid=flabels_test, include_values=True)
plt.savefig(os.path.join(result_path, 'confusion_matrix_values.png'))

# Plot the performance results and save
fig_acc = show_train_history(training_histories[best_model_desc], 'accuracy', 'val_accuracy')
plt.savefig(os.path.join(result_path, 'acc_plot.png'))
fig_los = show_train_history(history, 'loss', 'val_loss')
plt.savefig(os.path.join(result_path, 'loss_plot.png'))