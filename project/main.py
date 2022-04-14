from tensorflow.keras.utils import normalize
import numpy as np
import os
import sys
import cv2
import glob
import json

from PIL import Image
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm 

from model import *
from util import *
from u_node import UNode


HOME_DIR = os.getcwd()  # path to TuSimple dataset
run_name = 'run_toy'

# Specify paths to store performance plots and prediction masks
pred_path = os.path.join(HOME_DIR,'predicted_images')
if not os.path.isdir(pred_path):
    os.makedirs(pred_path, exist_ok=True)
result_path = './result/{0}'.format(run_name)
if not os.path.isdir(result_path):
    os.makedirs(result_path, exist_ok=True)

seed = 42
np.random.seed = seed

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
NUM_IMAGES = 100
EPOCHS = 20
NUM_FILTERS = 4

TRAIN_PATH = 'data/train_set/'
TEST_PATH = 'data/test_set/'

train_paths = glob.glob(TRAIN_PATH + 'labelled/images/*.png')
train_paths.sort()
train_paths = train_paths[:NUM_IMAGES]
train_labels = glob.glob(TRAIN_PATH + 'labelled/labels/*.png')
train_labels.sort()
test_paths = glob.glob(TEST_PATH + 'labelled/images/*.png')
test_paths.sort()
test_paths = test_paths[:NUM_IMAGES]
test_labels = glob.glob(TEST_PATH + 'labelled/labels/*.png')
test_labels.sort()


# Train data
X_train = np.zeros((len(train_paths), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_labels), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Resizing training images and masks')

for n, path in tqdm(enumerate(train_paths), total=len(train_paths)):   
    img = imread(path)[:,:,:IMG_CHANNELS]  
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img  #Fill empty X_train with values from img

for n, path in tqdm(enumerate(train_labels), total=len(train_labels)):
    mask = imread(path)
    mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                  preserve_range=True), axis=-1)  
        
    Y_train[n] = mask 

# X_train = np.array([imread(path)[:,:,:IMG_CHANNELS] for path in train_paths])
# Y_train = np.array([np.expand_dims(imread(path), axis=-1) for path in train_labels])

print(f"Train data type is {X_train.dtype}")

# Test data
X_test = np.zeros((len(test_paths), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_test = np.zeros((len(test_labels), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.bool)

for n, path in tqdm(enumerate(test_paths), total=len(test_paths)):
    img = imread(path)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

for n, path in tqdm(enumerate(test_labels), total=len(test_labels)):
    mask = imread(path)
    mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                  preserve_range=True), axis=-1)  
        
    Y_test[n] = mask 

# X_test = np.array([imread(path)[:,:,:IMG_CHANNELS] for path in test_paths])
# Y_test = np.array([np.expand_dims(imread(path), axis=-1) for path in test_labels])

print(f"Test data type is {X_test.dtype}")


# Print data shapes
print("Train data shape is: ", X_train.shape)
print("Train label shape is: ", Y_train.shape)
print("Max pixel value in image is: ", X_train.max())
print("Labels in the mask are : ", np.unique(Y_train))

# Input dimension
input_dim = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


# Callbacks
checkpointer = ModelCheckpoint('saved_model/model_for_tusimple.h5', verbose=1, save_best_only=True,
                save_weights_only=True)
callbacks = [EarlyStopping(patience=2, monitor='val_loss'), checkpointer]

model = UNode(num_filters=NUM_FILTERS, input_dim=input_dim, solver='dopri8')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

results = model.fit(X_train, Y_train, validation_split=0.2, batch_size=2, epochs=EPOCHS, callbacks=callbacks)

#plot the training and validation accuracy at each epoch
fig = plot_performance(results)
plt.savefig(os.path.join(result_path, 'performance_plot.png'))

#save prediction masks
for i in range(100):
    _, filename = os.path.split(test_paths[i])
    test_img = X_test[i]
    ground_truth=Y_test[i]
    test_img_input=np.expand_dims(test_img, 0)
    prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
    res = cv2.resize(prediction, dsize=(1280, 720), interpolation=cv2.INTER_CUBIC)
    plt.imshow(res, cmap='gray')
    plt.imsave(os.path.join(pred_path, filename), res, cmap='gray')