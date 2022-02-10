### Model imports 
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
from tensorflow.keras.optimizers import Adam, SGD, Adagrad
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.datasets import cifar100


