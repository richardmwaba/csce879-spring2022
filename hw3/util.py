import tensorflow as tf
import tensorflow_datasets as tfds
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
import numpy as np
from numpy.random import randn, randint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from skimage.transform import resize
from scipy.linalg import sqrtm
import os


def load_real_data():
    """Load and prepare cifar10 training images
    Argument:
        None
    Returns:
        Train (x50,000), test (x10,000) images 
    """
    (train_ds, _), (test_ds, _) = tf.keras.datasets.cifar10.load_data()
    # convert from unsigned ints to floats
    X_train, X_test = train_ds.astype('float32'), test_ds.astype('float32')
    # scale from [0,255] to [-1,1] in order to use tanh activation function in the generator
    X_train = (X_train - 127.5) / 127.5
    X_test = (X_test - 127.5) / 127.5
    return X_train, X_test


def generate_real_data(dataset, n_samples):
    """select a random subsample of real images
    Argument:
        dataset: name of dataset loaded from load_real_data()
        n_samples: number of images to be generated
    Returns:
        n_samples random real images and its corresponding labels
    """
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (all 1)
    y = np.ones((n_samples, 1))
    return X, y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_fake_data(g_model, latent_dim, n_samples):
    """use the generator to generate fake images, with class labels
    Argument:
        g_model: generator model
        latent_dim: size of the latent space
        n_samples: number of images to be generated
    Returns:
        n_samples fake images and its corresponding labels
    """
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (all 0)
    y = np.zeros((n_samples, 1))
    return X, y


def save_plot(examples, epoch, resultpath, n=10):
    """save generated fake images
    Argument:
        examples: collection of fake images
        epoch: epoch at which this function is called
        resultpath: path for saving plots
        n: number of images to be saved
    Returns:
        None
    """
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    # plot images
    for i in range(n):
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i])
        # save plot to file
        filename = 'generated_plot_e%03d_%02d.png' % (epoch+1, i+1)
        plt.savefig(os.path.join(resultpath,filename))
        plt.close()


def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, resultpath, n_samples=150):
    """evaluate the discriminator, plot generated images, save generator model
    Argument:
        epoch: epoch at which this function is called
        g_model: generator model
        d_model: discriminator model
        dataset: name of dataset loaded from load_real_data()
        latent_dim: size of the latent space
        resultpath: path for saving plots
        n_samples: number of images to be generated (both fake and real)
    Returns:
        None
    """
    # prepare real samples
    X_real, y_real = generate_real_data(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_data(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    # save plot
    save_plot(x_fake, epoch, resultpath)
    # save the generator model tile file
    filename = 'generator_model_%03d.h5' % (epoch+1)
    g_model.save(os.path.join(resultpath,filename))
    
    
def scale_images(images, new_shape):
    """scale an array of images to a new size
    Argument:
        images: collection of images
        new_shape: dim of output images
    Returns:
        Scaled images
    """
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        images_list.append(new_image)
    return np.asarray(images_list)


def calculate_fid(model, images1, images2):
    """calculate frechet inception distance (FID) for evaluation
    Argument:
        model: insert inception model here
        images1: first collection of images
        images2: second collection of images
    Returns:
        fid score
    """
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def evaluate_gan(real_imgs, fake_imgs):
    """evaluate performance of gan model by comparing real images and generated fake images
    Argument:
        real_imgs: collection of real images
        fake_imgs: collection of fake images
    Returns:
        fid score
    """
    # prepare the inception v3 model
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    # convert integer to floating point values
    real_imgs = real_imgs.astype('float32')
    fake_imgs = fake_imgs.astype('float32')
    # resize images
    real_imgs = scale_images(real_imgs, (299,299,3))
    fake_imgs = scale_images(fake_imgs, (299,299,3))
    # pre-process images
    real_imgs = preprocess_input(real_imgs)
    fake_imgs = preprocess_input(fake_imgs)
    
    fid = calculate_fid(model, real_imgs, fake_imgs)
    return fid