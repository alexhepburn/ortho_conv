# Utility functions for orthno conv and marginal Gaussianisation

import math
import numpy as np
from itertools import product

import tensorflow as tf

__all__ = ['im2toepidx', 'padding', 'get_sparse_toeplitz', 'get_toeplitz_idxs']


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

  return [pad_top, pad_bottom, pad_left, pad_right]


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