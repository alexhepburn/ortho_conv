import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import sys
import PIL.Image

import scipy.stats as stats
from ortho_conv.orthoconv.rbig import Conv_RBIG_Block_Inverse

from ortho_conv.orthoconv.gaussianisation import HistogramGaussianisation


# Load CIFAR-10
NNN = 10000
(train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()
train_images = train_images.astype('float32')
train_images = (train_images/127.5)-1 
train_images = train_images[0:NNN,:,:,:]

# PARAMS
epochs = 1
l1_act = 10e-5
params = {
    'block1':{
        'fw': 9,
        'stride': 1,
        'n_layers': 2,
        'epochs': epochs,
        'l1_act': l1_act,
        'l1_w': 0.0,
        'n_channel_factor': 1
    },
    'block2':{
        'fw': 5,
        'stride': 2,
        'n_layers': 2,
        'epochs': epochs,
        'l1_act': l1_act,
        'l1_w': 0.0,
        'n_channel_factor': 1
    },
    'block3':{
        'fw': 3,
        'stride': 4,
        'n_layers': 2,
        'epochs': 1,
        'l1_act': l1_act,
        'l1_w': 0.0,
        'n_channel_factor': 1
    },
    'block4':{
        'fw': 3,
        'stride': 8,
        'n_layers': 2,
        'epochs': 1,
        'l1_act': l1_act,
        'l1_w': 0.0,
        'n_channel_factor': 1
    },
    'block5':{
        'fw': 2,
        'stride': 12,
        'n_layers': 2,
        'epochs': 1,
        'l1_act': l1_act,
        'l1_w': 0.0,
        'n_channel_factor': 1
    },
    'block6':{
        'fw': 1,
        'stride': 18,
        'n_layers': 2,
        'epochs': 1,
        'l1_act': l1_act,
        'l1_w': 0.0,
        'n_channel_factor': 1
    }
}

# Store transformations
transformations_MG = []
transformations_LIN = []
log_pdf = []

# Gaussianise first
mg_1 = HistogramGaussianisation()
Z_G_small = mg_1.fit(train_images,n_samples=NNN)
train_images_T = mg_1.predict(train_images)
transformations_MG.append(mg_1)

# BLOCK 1
print('Block 1')
transformations_MG_aux, transformations_LIN_aux, train_images_T, \
    log_pdf_gauss_aux = Conv_RBIG_Block_Inverse(
        train_images_T,
        params['block1']['fw'],
        params['block1']['n_layers'],
        params['block1']['stride'],
        params['block1']['epochs'],
        params['block1']['l1_act'],
        params['block1']['l1_w'],
        params['block1']['n_channel_factor'],
        NNN=NNN)
transformations_MG.append(transformations_MG_aux)
transformations_LIN.append(transformations_LIN_aux)
log_pdf.append(np.mean(log_pdf_gauss_aux))

# BLOCK 2
print('Block 2')
transformations_MG_aux, transformations_LIN_aux, train_images_T, \
    log_pdf_gauss_aux = Conv_RBIG_Block_Inverse(
        train_images_T,
        params['block2']['fw'],
        params['block2']['n_layers'],
        params['block2']['stride'],
        params['block2']['epochs'],
        params['block2']['l1_act'],
        params['block2']['l1_w'],
        params['block2']['n_channel_factor'],
        NNN=NNN)
transformations_MG.append(transformations_MG_aux)
transformations_LIN.append(transformations_LIN_aux)
log_pdf.append(np.mean(log_pdf_gauss_aux))

# BLOCK 3
print('Block 3')
transformations_MG_aux, transformations_LIN_aux, train_images_T, \
    log_pdf_gauss_aux = Conv_RBIG_Block_Inverse(
        train_images_T,
        params['block3']['fw'],
        params['block3']['n_layers'],
        params['block3']['stride'],
        params['block3']['epochs'],
        params['block3']['l1_act'],
        params['block3']['l1_w'],
        params['block3']['n_channel_factor'],
        NNN=NNN)
transformations_MG.append(transformations_MG_aux)
transformations_LIN.append(transformations_LIN_aux)
log_pdf.append(np.mean(log_pdf_gauss_aux))

# BLOCK 4
print('Block 4')
transformations_MG_aux, transformations_LIN_aux, train_images_T, \
    log_pdf_gauss_aux = Conv_RBIG_Block_Inverse(
        train_images_T,
        params['block4']['fw'],
        params['block4']['n_layers'],
        params['block4']['stride'],
        params['block4']['fw'],
        params['block4']['epochs'],
        params['block4']['l1_act'],
        params['block4']['l1_w'],
        params['block4']['n_channel_factor'],
        NNN=NNN)
transformations_MG.append(transformations_MG_aux)
transformations_LIN.append(transformations_LIN_aux)
log_pdf.append(np.mean(log_pdf_gauss_aux))

# BLOCK 5
print('Block 5')
transformations_MG_aux, transformations_LIN_aux, train_images_T, \
    log_pdf_gauss_aux = Conv_RBIG_Block_Inverse(
        train_images_T,
        params['block5']['fw'],
        params['block5']['n_layers'],
        params['block5']['stride'],
        params['block5']['epochs'],
        params['block5']['l1_act'],
        params['block5']['l1_w'],
        params['block5']['n_channel_factor'],
        NNN=NNN)
transformations_MG.append(transformations_MG_aux)
transformations_LIN.append(transformations_LIN_aux)
log_pdf.append(np.mean(log_pdf_gauss_aux))

# BLOCK 6
print('Block 6')
transformations_MG_aux, transformations_LIN_aux, train_images_T, \
    log_pdf_gauss_aux = Conv_RBIG_Block_Inverse(
        train_images_T,
        params['block6']['fw'],
        params['block6']['n_layers'],
        params['block6']['stride'],
        params['block6']['epochs'],
        params['block6']['l1_act'],
        params['block6']['l1_w'],
        params['block6']['n_channel_factor'],
        NNN=NNN)
transformations_MG.append(transformations_MG_aux)
transformations_LIN.append(transformations_LIN_aux)
log_pdf.append(np.mean(log_pdf_gauss_aux))
