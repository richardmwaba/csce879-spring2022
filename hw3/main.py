import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from model import *
from util import *


# Set configurations and hyperparameters
run_name = 'run_toy'
n_epochs = 100
batch_size = 128
latent_dim = 100  # size of the latent space
n_epochs_save = 10  # save model every x epochs
n_samples = 1000  # size of the two fake and real image collections for evaluation
patience = 5  # for early stopping

# Set path to save performance plots
resultpath = './result/{0}'.format(run_name)
if not os.path.isdir(resultpath):
    os.makedirs(resultpath, exist_ok=True)

# create discriminator
d_model = discriminator()
# create generator
g_model = generator(latent_dim)
# create gan
gan_model = gan(g_model, d_model)
# load image data
train_ds, test_ds = load_real_data()  # shape (n_sample, width, height, channels)

# train model
bat_per_epo = int(train_ds.shape[0] / batch_size)
half_batch = int(batch_size / 2)  # first half for real, second half for fake samples
fid_record = []
for i in range(n_epochs):
    for j in range(bat_per_epo):  # enumerate batches over the training set
        # discriminator model is updated twice per batch, once with real samples and once with fake samples
        # get randomly selected 'real' samples
        X_real, y_real = generate_real_data(train_ds, half_batch)
        # update discriminator model weights
        d_model.train_on_batch(X_real, y_real)  # run a single gradient update on a single batch of data
        # generate 'fake' examples
        X_fake, y_fake = generate_fake_data(g_model, latent_dim, half_batch)
        # update discriminator model weights
        d_model.train_on_batch(X_fake, y_fake)
        # prepare points in latent space as input for the generator
        X_gan = generate_latent_points(latent_dim, batch_size)
        # create inverted labels for the fake samples
        y_gan = np.ones((batch_size, 1))
        # update the generator via the discriminator's error
        gan_model.train_on_batch(X_gan, y_gan)
    
    # summarize performance and save generated images after certain number of epochs
    if (i+1) % n_epochs_save == 0:
        summarize_performance(i, g_model, d_model, train_ds, latent_dim, resultpath)
    
    # evaluate using frechet inception distance (FID)
    fake_imgs, _ = generate_fake_data(g_model, latent_dim, n_samples)
    real_imgs, _ = generate_real_data(test_ds, n_samples)
    fid = evaluate_gan(real_imgs, fake_imgs)
    print('Epoch {}, fid={}'.format(i+1, fid))
    fid_record.append(fid)
    if (i+1) > patience:
        recent_fids = fid_record[-patience-1:]
        if all(recent_fids[i] <= recent_fids[i + 1] for i in range(len(recent_fids) - 1)):  # early stopping
            summarize_performance(i, g_model, d_model, train_ds, latent_dim, resultpath)
            break
