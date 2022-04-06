from tensorflow.keras.utils import normalize
import os
import sys
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import glob
import json

from model import *
from util import *


HOME_DIR = '/common/appliedmath/nthach17/TUSimple/'  # path to TuSimple dataset
run_name = 'run_toy'

# Specify paths to store performance plots and prediction masks
pred_path = os.path.join(HOME_DIR,'predicted_image')
if not os.path.isdir(pred_path):
    os.makedirs(pred_path, exist_ok=True)
result_path = './result/{0}'.format(run_name)
if not os.path.isdir(result_path):
    os.makedirs(result_path, exist_ok=True)

image_directory = os.path.join(HOME_DIR, 'original_image')
mask_directory = os.path.join(HOME_DIR, 'label_image')    

if not (os.path.isdir(image_directory) and os.path.isdir(mask_directory)):
    process(HOME_DIR, 'label_data_0313.json')
    process(HOME_DIR, 'label_data_0531.json')
    process(HOME_DIR, 'label_data_0601.json')

num_images = len(os.listdir(image_directory))

image_names = glob.glob(image_directory + '/*.png')
image_names.sort()
image_names_subset = image_names[0:num_images]
images = [cv2.imread(img, 0) for img in image_names_subset]
image_dataset = np.array(images)
image_dataset = np.expand_dims(image_dataset, axis = 3)

mask_names = glob.glob(mask_directory + '/*.png')
mask_names.sort()
mask_names_subset = mask_names[0:num_images]
masks = [cv2.imread(mask, 0) for mask in mask_names_subset]
mask_dataset = np.array(masks)
mask_dataset = np.expand_dims(mask_dataset,axis = 3)

print("Image data shape is: ", image_dataset.shape)
print("Mask data shape is: ", mask_dataset.shape)
print("Max pixel value in image is: ", image_dataset.max())
print("Labels in the mask are : ", np.unique(mask_dataset))

#Normalize images
image_dataset = image_dataset /255.  #Can also normalize or scale using MinMax scaler
#Do not normalize masks, just rescale to 0 to 1.
mask_dataset = mask_dataset /255.  #PIxel values will be 0 or 1

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(image_dataset,
                                                                         mask_dataset,
                                                                         range(len(image_dataset)),
                                                                         test_size = 0.20, random_state = 42)

IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = build_unet(input_shape, n_classes=1)
model.compile(optimizer=Adam(learning_rate = 1e-3), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=10, 
                    validation_data=(X_test, y_test),              
                    shuffle=False)

#plot the training and validation accuracy at each epoch
fig = plot_performance(history)
plt.savefig(os.path.join(result_path, 'performance_plot.png'))

#save prediction masks
for i in range(len(X_test)):
    path, filename = os.path.split(image_names[idx_test[i]])
    test_img = X_test[i]
    ground_truth=y_test[i]
    test_img_input=np.expand_dims(test_img, 0)
    prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
    res = cv2.resize(prediction, dsize=(1280, 720), interpolation=cv2.INTER_CUBIC)
    plt.imshow(res, cmap='gray')
    plt.imsave(os.path.join(pred_path,filename), res, cmap='gray')