import tensorflow as tf
import numpy as np

from tqdm import tqdm
from model_MNIST import *
from util_MNIST import *

train_ds = load_train('fashion_mnist')
validation_ds = load_valid('fashion_mnist')
model, optimizer = model1()

loss_values = []
train_accuracy_values = []

for epoch in range(3):
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
test_true = np.array([])
test_pred = np.array([])

# Loop through one epoch of data
for epoch in range(10):
    for batch in tqdm(validation_ds):

        x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])/255
        labels = batch['label']
        logits = model(x)

        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
        true = tfds.as_numpy(labels)
        pred = tfds.as_numpy(predictions)
        test_true = np.concatenate([test_true, true])
        test_pred = np.concatenate([test_pred, pred])
        vali_accuracy_values.append(accuracy)
    print("Accuracy:", np.mean(vali_accuracy_values))
print(model.summary())
    
# accuracy
print("Accuracy:", np.mean(vali_accuracy_values))

print(tf.math.confusion_matrix(test_true, test_pred))
