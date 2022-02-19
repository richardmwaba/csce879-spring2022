from __future__ import print_function
import tensorflow as tf
import numpy as np
from cmath import inf
from gc import callbacks
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.callbacks import EarlyStopping
from model_MNIST import *
from util_MNIST import *
import os



images_train, labels_train, images_valid, labels_valid, images_test, labels_test = load_data('fashion_mnist')
model = model12()
#model = eval(model_name)(input_dim, nclass)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(
    images_train, labels_train,
    batch_size=32,
    epochs=50,
    verbose=1,
    validation_data=(images_valid, labels_valid),
    callbacks= [EarlyStopping(monitor='val_accuracy', patience=5)]
    )

train_accuracy = history.history['accuracy'][-1]
print('train_acc',train_accuracy)
val_accuracy = history.history['val_accuracy'][-1]
print('val_acc',val_accuracy)
train_loss = history.history['loss'][-1]
print('train_loss',train_loss)
val_loss = history.history['val_loss'][-1]
print('val_loss',val_loss)
print(model.summary())

test_loss, test_acc = model.evaluate(images_test,  labels_test, verbose=1)
print('\nTest accuracy:', test_acc)

fig_acc = show_acc_history(history.history['accuracy'], history.history['val_accuracy'])
plt.savefig(os.path.join('./hw1', 'model12_acc.png'))
fig_loss = show_loss_history(history.history['loss'],history.history['val_loss'])
plt.savefig(os.path.join('./hw1', 'model12_loss.png'))
fig_cm = c_m(images_test, model, labels_test)
plt.savefig(os.path.join('./hw1', 'model12_cm.png'))

print('confidence interval',confidence_interval(test_acc))
