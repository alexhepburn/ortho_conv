# Convolutional RBIG functions

import tensorflow as tf
import numpy as np

import scipy.stats as stats

from ortho_conv.orthoconv.gaussianisation import HistogramGaussianisation
from ortho_conv.orthoconv.ortho_conv import OrthoConv
from ortho_conv.orthoconv.utils import SpatialReshape

__all__ = ['Conv_RBIG_Block_Inverse']


def Conv_RBIG_Block_Inverse(train_images_T,
                    fw,
                    n_layers=2,
                    stride=1,
                    epochs=10,
                    l1_act=0.0,
                    l1_w=0.0,
                    n_channel_factor=1.0,
                    NNN=8000):
    """
    Function for returning the transformations of 1 block of convolutional
    RBIG using the inverse rotation after Gaussianisation and spatial reshaping.
    """
    transformations_MG = []
    transformations_LIN = []

    print('Layer: ', 0)
    # ORTHO
    reshape = SpatialReshape(stride=stride)
    train_images_T = reshape.predict(train_images_T)
    Conv_L1 = OrthoConv(filter_width=fw,
                        stride=1,
                        input_channels = train_images_T.shape[3],
                        n_channel_factor=n_channel_factor)
    Conv_L1.fit(
        np.random.permutation(train_images_T)[0:NNN,:,:,:],
        epochs=epochs)
    train_images_T = Conv_L1.predict(train_images_T)
    transformations_LIN.append(reshape)
    transformations_LIN.append(Conv_L1)

    # MG
    mg_1 = HistogramGaussianisation()
    Z_G_small = mg_1.fit(train_images_T,n_samples=NNN)
    train_images_T = mg_1.predict(train_images_T)
    transformations_MG.append(mg_1)

    # Take the inverse
    train_images_T = Conv_L1.inverse(train_images_T)

    for n in range(1, n_layers):
        print('Layer: ', n)
         
        # ORTHO
        Conv_L1 = OrthoConv(filter_width=fw,
                            stride=1,
                            input_channels=train_images_T.shape[3],
                            n_channel_factor=n_channel_factor)
        Conv_L1.fit(
            np.random.permutation(train_images_T)[0:NNN,:,:,:],
            epochs=epochs,
            activation_L1=l1_act,
            weight_L1 = l1_w)
        train_images_T = Conv_L1.predict(train_images_T) # predict
        transformations_LIN.append(Conv_L1)

        # MG
        mg_1 = HistogramGaussianisation()
        Z_G_small = mg_1.fit(train_images_T,n_samples=NNN)
        train_images_T = mg_1.predict(train_images_T)
        transformations_MG.append(mg_1)

        train_images_T = Conv_L1.inverse(train_images_T)
    
    aux = np.random.permutation(
        np.reshape(train_images_T,
                   (np.prod(train_images_T.shape[0:-1]),
                    train_images_T.shape[3])))[0:100000,:]
    mm = np.mean(aux,axis=0)
    cc = np.cov(aux.T)
    
    log_pdf_gauss_aux = stats.multivariate_normal.logpdf(aux,mm,cc)

    train_images_T = reshape.inverse(train_images_T)

    return (transformations_MG, transformations_LIN, train_images_T,
        log_pdf_gauss_aux)