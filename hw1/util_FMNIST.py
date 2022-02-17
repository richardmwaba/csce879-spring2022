import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
DATA_DIR = './tensorflow-datasets/'

def load_train(dataset):

    train = tfds.load(dataset, split='train[:90%]', data_dir=DATA_DIR)
    train_ds = train.shuffle(1024).batch(32)
    
    return train_ds

def load_valid(dataset):
    
    validation = tfds.load(dataset, split='train[-10%:]', data_dir=DATA_DIR)
    validation_ds = validation.shuffle(1024).batch(32)
    
    return validation_ds
