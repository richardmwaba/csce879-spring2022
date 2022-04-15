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
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm 

from model import UNet
from util import *


HOME_DIR = os.getcwd()  # path to TuSimple dataset
run_name = 'run_toy'

# Specify paths to store performance plots and prediction masks
result_path = os.path.join(HOME_DIR, f'result/{run_name}')
if not os.path.isdir(result_path):
    os.makedirs(result_path, exist_ok=True)

pred_path = os.path.join(result_path,'predicted_images')
if not os.path.isdir(pred_path):
    os.makedirs(pred_path, exist_ok=True)

saved_model_path = os.path.join(result_path, 'saved_model')
if not os.path.isdir(saved_model_path):
    os.makedirs(saved_model_path, exist_ok=True)

seed = 42
np.random.seed = seed

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
NUM_IMAGES = 1000
EPOCHS = 100
NUM_FILTERS = 32
NUM_CLASSES = 6

TRAIN_PATH = 'data/train_set/'
TEST_PATH = 'data/test_set/'

train_paths = glob.glob(TRAIN_PATH + 'labelled/images/*.png')
train_paths.sort()
# train_paths = train_paths[:NUM_IMAGES]
train_labels = glob.glob(TRAIN_PATH + 'labelled/labels/*.png')
train_labels.sort()
test_paths = glob.glob(TEST_PATH + 'labelled/images/*.png')
test_paths.sort()
# test_paths = test_paths[:NUM_IMAGES]
test_labels = glob.glob(TEST_PATH + 'labelled/labels/*.png')
test_labels.sort()
# test_labels = test_labels[:NUM_IMAGES]


# Train data
# X_train = np.zeros((len(train_paths), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
# Y_train = np.zeros((len(train_labels), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
print('Resizing training images and masks')

X_train = []
for n, path in tqdm(enumerate(train_paths), total=len(train_paths)):   
    img = imread(path)[:,:,:IMG_CHANNELS]  
    # img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_NEAREST)
    X_train.append(img)
X_train = np.array(X_train)

Y_train = []
for n, path in tqdm(enumerate(train_labels), total=len(train_labels)):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_NEAREST)  
        
    Y_train.append(mask) 
Y_train = np.array(Y_train)

# Encode train labels
labelencoder = LabelEncoder()
n, h, w = Y_train.shape
Y_train_reshaped = Y_train.reshape(-1,1)
Y_train_encoded = labelencoder.fit_transform(Y_train_reshaped)
Y_train = Y_train_encoded.reshape(n, h, w)
Y_train = np.expand_dims(Y_train, axis=3)

# Convert to categorical
Y_train_cat = to_categorical(Y_train, num_classes=NUM_CLASSES)


print(f"Train data type is {X_train.dtype}")

# Test data
# X_test = np.zeros((len(test_paths), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
# Y_test = np.zeros((len(test_labels), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

X_test = []
for n, path in tqdm(enumerate(test_paths), total=len(test_paths)):
    img = imread(path)[:,:,:IMG_CHANNELS]
    # img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_NEAREST)
    X_test.append(img)
X_test = np.array(X_test)

Y_test = []
for n, path in tqdm(enumerate(test_labels), total=len(test_labels)):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_NEAREST)
        
    Y_test.append(mask)
Y_test = np.array(Y_test) 

# Encode test labels
labelencoder = LabelEncoder()
n, h, w = Y_test.shape
Y_test_reshaped = Y_test.reshape(-1,1)
Y_test_encoded = labelencoder.fit_transform(Y_test_reshaped)
Y_test = Y_test_encoded.reshape(n, h, w)
Y_test = np.expand_dims(Y_test, axis=3)

# Convert to categorical
Y_test_cat = to_categorical(Y_test, num_classes=NUM_CLASSES)


print(f"Test data type is {X_test.dtype}")


# Print data shapes
print("Train data shape is: ", X_train.shape)
print("Train label shape is: ", Y_train.shape)
print("Max pixel value in image is: ", X_train.max())
print("Labels in the mask are : ", np.unique(Y_train))

# Input dimension
input_dim = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


# Callbacks
checkpointer = ModelCheckpoint(f'{saved_model_path}/model_for_tusimple.h5', verbose=1, save_best_only=True,
                save_weights_only=True)
callbacks = [EarlyStopping(patience=2, monitor='val_loss'), checkpointer]

model = UNet(input_shape=input_dim, filters=NUM_FILTERS, num_classes=NUM_CLASSES)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

results = model.fit(X_train, Y_train_cat, validation_split=0.2, batch_size=32, epochs=EPOCHS, callbacks=callbacks)

#plot the training and validation accuracy at each epoch
fig = plot_performance(results)
plt.savefig(os.path.join(result_path, 'performance_plot.png'))

#save prediction masks
for i in range(len(X_test)):
    _, filename = os.path.split(test_paths[i])
    test_img = X_test[i]
    img = test_img.reshape(1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    ground_truth=Y_test_cat[i]
    prediction = model.predict(img)
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]
    # res = cv2.resize(prediction, dsize=(1280, 720), interpolation=cv2.INTER_CUBIC)
    plt.imshow(predicted_img, cmap='gray')
    plt.imsave(os.path.join(pred_path, filename), predicted_img)