import os
import os.path

import sys
import warnings
import copy
from glob import glob
import argparse

import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
# from tensorflow.contrib.layers.python.layers import initializers

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from util import dice, focal_tversky, tversky
# from loss import discriminative_loss
from data_generator import get_all_images
from model import UNet



def run():
    parser = argparse.ArgumentParser()
    # Directories
    parser.add_argument('-s','--srcdir', default='data', help="Source directory of TuSimple dataset")
    parser.add_argument('-m', '--modeldir', default='pretrained_model', help="DIrectory for pretrained model")
    parser.add_argument('-o', '--outdir', default='saved_model', help="Directory for trained model")
    # parser.add_argument('-l', '--logdir', default='log', help="Log directory for tensorboard and evaluation files")
    # Hyperparameters
    # parser.add_argument('--epochs', type=int, default=50, help="Number of epochs")
    # parser.add_argument('--var', type=float, default=1., help="Weight of variance loss")
    # parser.add_argument('--dist', type=float, default=1., help="Weight of distance loss")
    # parser.add_argument('--reg', type=float, default=0.001, help="Weight of regularization loss")
    # parser.add_argument('--dvar', type=float, default=0.5, help="Cutoff variance")
    # parser.add_argument('--ddist', type=float, default=1.5, help="Cutoff distance")

    args = parser.parse_args()

    if not os.path.isdir(args.srcdir):
        raise IOError(f'Directory: {args.srcdir} does not exist')
    if not os.path.isdir(args.modeldir):
        raise IOError(f'Directory: {args.modeldir} does not exist')
    # if not os.path.isdir(args.logdir):
    #     os.mkdir(args.logdir)

    image_shape = (512, 512)
    data_dir = args.srcdir #os.path.join('.', 'data')
    model_dir = args.modeldir
    output_dir = args.outdir
    # log_dir = args.logdir

    image_paths = glob(os.path.join(data_dir, 'images', '*.png'))
    label_paths = glob(os.path.join(data_dir, 'labels', '*.png'))

    image_paths.sort()
    label_paths.sort()
    
    image_paths = image_paths[0:10]
    label_paths = label_paths[0:10]

    images, labels = get_all_images(image_shape, image_paths, label_paths)

    X_train, X_valid, y_train, y_valid = train_test_split(images, labels, test_size=0.20, random_state=42)

    print('Number of train samples', len(y_train))
    print('Number of valid samples', len(y_valid))
    print('Shape of image is: ', X_train[0].shape)

    model = UNet()
    model.create_model(input_size=[512,512,3],loss=focal_tversky,metrics=dice)


    model.train_model(filepath='saved_model', X_train=X_train, y_train=y_train, 
                    X_val=X_valid, y_val=y_valid, epochs=20, display_callback=None)

    # train_model.evaluate(X_test,y_test)
    model.save_model("path_to_dest.tf")



if __name__ == '__main__':
    run()