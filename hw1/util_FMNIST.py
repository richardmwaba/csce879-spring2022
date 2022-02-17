import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
DATA_DIR = './tensorflow-datasets/'


train = tfds.load('fashion_mnist', split='train[:90%]', data_dir=DATA_DIR)
validation = tfds.load('fashion_mnist', split='train[-10%:]', data_dir=DATA_DIR)
train_ds = train.shuffle(1024).batch(32)
validation_ds = validation.shuffle(1024).batch(32)
