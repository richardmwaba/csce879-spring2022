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

trained_models = {}
training_histories = {}

input_dim = [28, 28, 1]
model_name = ['model1']
nclass = 10

images_train, labels_train, images_valid, labels_valid, images_test, labels_test = load_data('fashion_mnist')
#model, optimizer = model1()
model = eval(model_name)(input_dim, nclass, input_x=images_train)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(
    images_train, labels_train,
    batch_size=32,
    epochs=100,
    verbose=1,
    validation_data=(images_valid, labels_valid),
    callbacks= [EarlyStopping(monitor='val_accuracy', patience=5)]
    )
final_accuracy = history.history['val_accuracy'][-1]
print(model.summary())


fig_acc = show_train_history(history.history['accuracy'], history.history['val_accuracy'])
plt.savefig(os.path.join(os.getcwd(), 'acc.png'))
fig_loss = show_train_history(history.history['loss'],history.history['val_loss'])
plt.savefig(os.path.join(os.getcwd(), 'loss.png'))

print(tf.math.confusion_matrix(test_true, test_pred))
