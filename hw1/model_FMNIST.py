import tensorflow as tf
from keras.models import Sequential
from util_MNIST import *
from keras.layers import Dense, Flatten, Input

def widemodel():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(256, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, tf.nn.softmax))
    
    return model


def medmodel():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, tf.nn.softmax))
    
    return model
def deepmodel():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(64, tf.nn.relu))
    model.add(tf.keras.layers.Dense(64, tf.nn.relu))
    model.add(tf.keras.layers.Dense(64, tf.nn.relu))
    model.add(tf.keras.layers.Dense(64, tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, tf.nn.softmax))
    
    return model

class L2DenseNetwork(tf.Module):
    def __init__(self, name=None):
        super(L2DenseNetwork, self).__init__(name=name) # remember this call to initialize the superclass
        self.dense_layer1 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.dense_layer2 = tf.keras.layers.Dense(10)
        
    def l2_loss(self):
        # Make sure the network has been called at least once to initialize the dense layer kernels
        return tf.nn.l2_loss(self.dense_layer1.kernel) + tf.nn.l2_loss(self.dense_layer2.kernel)

    @tf.function
    def __call__(self, x):
        embed = self.dense_layer1(x)
        output = self.dense_layer2(embed)
        return output
    
# Defining, creating and calling the network repeatedly will trigger a WARNING about re-tracing the function
# So we'll check to see if the variable exists already
if 'l2_dense_net' not in locals():
    l2_dense_net = L2DenseNetwork()
l2_dense_net(tf.ones([1, 100]))

l2_loss = l2_dense_net.l2_loss()                     # calculate l2 regularization loss
cross_entropy_loss = 0.                              # calculate the classification loss
total_loss = cross_entropy_loss + L2_COEFF * l2_loss # and add to the total loss, then calculate gradients
