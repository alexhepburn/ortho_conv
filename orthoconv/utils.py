# Utility functions for orthno conv and marginal Gaussianisation

import itertools
import math
import numpy as np
from itertools import product

import tensorflow as tf

__all__ = ['SpatialReshape', 'im2toepidx', 'padding', 'get_sparse_toeplitz',
           'get_toeplitz_idxs', 'get_conv_square_ar_mask', 'get_conv_weight_np']


class SpatialReshape:
    """
    A class to rearange neighbouring pixels into the channels.
    
    Will take every nth pixel (where n is the stride set) and add it in the
    channel dimension.
    
    Parameters
    ----------
    stride : int (default=2)
        
    """
    def __init__(self, stride=2):
        self.stride = stride

    def predict(self,
                images : np.ndarray) -> np.ndarray:
        """ The forward pass of the transformation.

        Parameters
        ----------
        images : numpy.ndarray
            The images to transform. Must be of shape `[N, H, W, C]`.
        """
        self.in_shape = images.shape # For use on the inverse.
        output_height, output_width = images.shape[1]//self.stride, \
            images.shape[2]//self.stride
        output_channels = self.stride**2 * images.shape[-1]

        out_images = np.zeros((images.shape[0], output_height, output_width,
                               output_channels), dtype=np.float32)
        
        # Need to crop the images so the [H, W] are multiples of the stride.
        images = images[:, 0:output_height*self.stride, 
                        0:output_width*self.stride, :]

        # Get indices needed
        self.indices = list(product(
            list(range(self.stride)), repeat=2))
        for n, (i, k) in enumerate(self.indices):
            out_images[:, :, :, n*images.shape[-1]:images.shape[-1]*(n+1)] = \
                images[:, i::self.stride, k::self.stride, :]

        return out_images

    def inverse(self,
                encoded : np.ndarray) -> np.ndarray:
        """ The inverse of the transformation.

        Parameters
        ----------
        encoded : numpy.ndarray
            The encoded images. Must be of shape `[N, H, W, C]`.
            
        Returns
        -------
        reconstructed : numpy.ndarray
            The reocnstruction of the images using the inverse transform.
        """
        # Effectively zero padding the images to be the same size as the input.
        reconstructed = np.zeros(self.in_shape)
        for n, (i, k) in enumerate(self.indices):
            reconstructed[:, i::self.stride, k::self.stride, :] = encoded[:, :,
                :, n*reconstructed.shape[-1]:reconstructed.shape[-1]*(n+1)]

        return reconstructed
    
    def logdetJ(self, input_shape):
        """Calculates log det(J)
        Calculates the logarithm of the determinant of the Jacobian of the
        transformation.
        """
        logsdetJ = tf.repeat(0.0, input_shape[0])
        return logsdetJ


def im2toepidx(c, i, j, h, w):
    return c*h*w + i*w + j


def padding(input_size, f_size, stride):
  h, w = input_size[0:2]
  out_height = math.ceil(float(h) / float(stride[0]))
  out_width  = math.ceil(float(w) / float(stride[1]))

  pad_h = max((out_height - 1) * stride[0] + f_size[0] - h, 0)
  pad_w = max((out_width - 1) * stride[1] + f_size[1] - w, 0)

  pad_top = pad_h // 2
  pad_bottom = pad_h - pad_top
  pad_left = pad_w // 2
  pad_right = pad_w- pad_left

  return [max(pad_top, pad_bottom), max(pad_left, pad_right)]


def get_sparse_toeplitz(f, dshape, T_idxs, f_idxs):
  t_size = (T_idxs.max(axis=0)[0] + 1, np.prod(dshape))
  vals = tf.gather(tf.reshape(f, [-1]), axis=0, indices=f_idxs)
  return tf.sparse.SparseTensor(T_idxs, vals, t_size)


def get_toeplitz_idxs(fshape, dshape, f_stride=(1,1)):
  assert fshape[1] == dshape[0], "data channels must match filters channels"
  fh, fw = fshape[-2:]
  ic, ih, iw = dshape
  s_pad = padding(dshape[1:], fshape[2:], stride=f_stride)
  oh = int(math.floor((ih + 2 * s_pad[0] - fh) / f_stride[0]) + 1)
  ow = int(math.floor((iw + 2 * s_pad[1] - fw) / f_stride[1]) + 1)
  oc = fshape[0]

  T_idxs = []
  f_idxs = []
  for outch, outh, outw in product(range(oc), range(oh), range(ow)):
    for fi, fj in product(range(0-s_pad[0], fh-s_pad[0]), range(0-s_pad[1], fw-s_pad[1])):
      readh, readw = (outh*f_stride[0]) + fi, (outw*f_stride[1]) + fj
      if readh < 0 or readw < 0 or readh >= ih or readw >= iw:
        continue
          
      for inch in range(ic):
        Mj = im2toepidx(inch, readh, readw, ih, iw)
        Mi = im2toepidx(outch, outh, outw, oh, ow)
        T_idxs.append([Mi, Mj])
        f_flat_idx = outch*(ic*fh*fw) + inch*(fh*fw) + (fi+s_pad[0])*fh + (fj+s_pad[1])
        f_idxs.append(f_flat_idx)

  return (np.array(T_idxs), np.array(f_idxs))


def get_conv_weight_np(filter_shape, stable_init=True, unit_testing=False):
    weight_np = np.random.randn(*filter_shape) * 0.02
    kcent = (filter_shape[0] - 1) // 2
    if stable_init or unit_testing:
        weight_np[kcent, kcent, :, :] += 1. * np.eye(filter_shape[3])
    weight_np = weight_np.astype('float32')
    return weight_np


def get_linear_ar_mask(n_in, n_out, zerodiagonal=False):
    assert n_in % n_out == 0 or n_out % n_in == 0, "%d - %d" % (n_in, n_out)

    mask = np.ones([n_in, n_out], dtype=np.float32)
    if n_out >= n_in:
        k = n_out // n_in
        for i in range(n_in):
            mask[i + 1:, i * k:(i + 1) * k] = 0
            if zerodiagonal:
                mask[i:i + 1, i * k:(i + 1) * k] = 0
    else:
        k = n_in // n_out
        for i in range(n_out):
            mask[(i + 1) * k:, i:i + 1] = 0
            if zerodiagonal:
                mask[i * k:(i + 1) * k:, i:i + 1] = 0
    return mask


def get_conv_square_ar_mask(h, w, n_in, n_out, zerodiagonal=False):
    """
    Function to get autoregressive convolution with square shape.
    """
    l = (h - 1) // 2
    m = (w - 1) // 2
    mask = np.ones([h, w, n_in, n_out], dtype=np.float32)
    mask[:l, :, :, :] = 0
    mask[:, :m, :, :] = 0
    mask[l, m, :, :] = get_linear_ar_mask(n_in, n_out, zerodiagonal)
    return mask
