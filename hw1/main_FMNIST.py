import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from model_MNIST import *
from util_MNIST import *
import os

train_ds = load_train('fashion_mnist')
validation_ds = load_valid('fashion_mnist')
model, optimizer = model1()

train_loss_values = []
valid_loss_values = []
train_loss_values_plot = []
valid_loss_values_plot = []
train_accuracy_plot = []
valid_accuracy_plot = []
train_accuracy_values = []
valid_accuracy_values = []
test_true = np.array([])
test_pred = np.array([])

for epoch in range(4):
    for batch in tqdm(train_ds):
        with tf.GradientTape() as tape:
            # run network
            x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])/255
            labels = batch['label']
            logits = model(x)

            # calculate loss
            train_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)    
        train_loss_values.append(train_loss)
    
        # gradient update
        grads = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
        # calculate accuracy
        predictions = tf.argmax(logits, axis=1)
        train_accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
        train_accuracy_values.append(train_accuracy)
        
        
    for batch in tqdm(validation_ds):
        x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])/255
        labels = batch['label']
        logits = model(x)
        valid_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)    
        valid_loss_values.append(valid_loss)

        predictions = tf.argmax(logits, axis=1)
        valid_accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
        true = tfds.as_numpy(labels)
        pred = tfds.as_numpy(predictions)
        test_true = np.concatenate([test_true, true])
        test_pred = np.concatenate([test_pred, pred])
        valid_accuracy_values.append(valid_accuracy)
        
        
    train_loss_values_plot.append(np.mean(np.concatenate(train_loss_values)))
    valid_loss_values_plot.append(np.mean(np.concatenate(valid_loss_values)))
    train_accuracy_plot.append(np.mean(train_accuracy_values))
    valid_accuracy_plot.append(np.mean(valid_accuracy_values))


print(model.summary())
# accuracy
print("Train Accuracy:", np.mean(train_accuracy_values))
print("Test Accuracy:", np.mean(valid_accuracy_values))
fig_acc = show_train_history(train_accuracy_plot, valid_accuracy_plot)
plt.savefig(os.path.join(os.getcwd(), 'acc.png'))
fig_loss = show_train_history(train_loss_values_plot,valid_loss_values_plot)
plt.savefig(os.path.join(os.getcwd(), 'loss.png'))

print(tf.math.confusion_matrix(test_true, test_pred))
