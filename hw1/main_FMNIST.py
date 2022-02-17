import tensorflow as tf
import numpy as np

from tqdm import tqdm
from model_MNIST import *
from util_MNIST import *

loss_values = []
train_accuracy_values = []

for epoch in range(5):
    for batch in tqdm(train_ds):
        with tf.GradientTape() as tape:
            # run network
            x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])/255
            labels = batch['label']
            logits = model(x)

            # calculate loss
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)    
        loss_values.append(loss)
    
        # gradient update
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
        # calculate accuracy
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
        train_accuracy_values.append(accuracy)
        
print(model.summary())
    
# accuracy
print("Accuracy:", np.mean(train_accuracy_values))

vali_accuracy_values = []

# Loop through one epoch of data
for epoch in range(1):
    for batch in tqdm(validation_ds):

        x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])/255
        labels = batch['label']
        logits = model(x)

        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
        vali_accuracy_values.append(accuracy)

print(model.summary())
    
# accuracy
print("Accuracy:", np.mean(vali_accuracy_values))
