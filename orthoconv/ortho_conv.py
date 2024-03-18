# Orthonormal convolution class

from typing import List, Union

import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors

import numpy as np

import orthoconv.utils as utils
from orthoconv.inverses.inverse_cython import Inverse

__all__ = ['OrthoConv', 'InvertibleConv2dEmerging']


class OrthoConv:
    """
    A class for orthonormal convolutions

    A 2-D convolution that maintains dimensionality and can be trained to be
    orthonormal and invertible. If we minimise a reconstruction loss, then

    .. math::
        V^T V \cdot \tilde{Im} = \tilde{Im}

    and the convolutional matrix `V` is orthonormal.

    Parameters
    ----------
    filter_width : int (default=3)
        Width (and height) of the convolutional filter. It is assumed taht the
        filter is square (filter height = filter width).
    stride : int (default=1)
        The stride of the sliding window in the convolution operation. This is
        assumed to be the same for both dimensions.
    input_channels : int (default=1)
        The number of channels in the input.

    Attributes
    ----------
    shape_W : List[int]
        The shape of the convolutional filter
    W : tf.Variable
        The convolution filter used of size `[filter_width`, filter_width, 
        input_channels, stride^2*input_channels]`. `W` is initialised to be
        a random orthogonal matrix.
    T_idxs : Union[List[int], None]
        A list of indices for the convolutional matrix corresponding to the
        convolution operation. This is set when called, and will be initialised
        as `None`.
    f_idxs : Union[List[int], None]
        A list of filter indices that corresponds to the indices in `T_idxs`.
        This is set when called, and will be initialised as `None`.
    

    """
    def __init__(self,
                 filter_width: int = 3,
                 stride: int = 1,
                 input_channels: int = 1) -> None:
        """
        Constructs an OrthoConv class.
        """
        self.stride_size = stride
        self.FW = filter_width
        self.N_ch = input_channels
        
        # loss
        self.loss_MSE = tf.keras.losses.MeanSquaredError()
        self.loss_MAE = tf.keras.regularizers.L1()
        # weights
        output_channels = stride**2 * input_channels
        self.shape_W = [filter_width, filter_width , input_channels,
                        output_channels]

        # Orthonormal initialization
        initializer = tf.keras.initializers.Orthogonal(gain=1.0)
        self.W = tf.Variable( initializer(shape=self.shape_W), trainable=True)
        
        self.T_idxs, self.f_idxs = None, None

    def fit(self,
            images : np.ndarray,
            BATCH : int = 16,
            learning_rate : float = 0.001,
            epochs : int = 2,
            activation_L1 : float = 10e-5,
            weight_L1 : float = 0.0) -> None:
        """Trains to model for reconstruction.

        Optimises the model using Adam optimiser for the L2 loss between the
        input and the reconstructed input (one convolution then transposed
        convolution).

        Parameters
        ----------
        images : numpy.ndarray
            The images to train the layer on. Must be of shape `[N, H, W, C]`.
        batch : int (default=16)
            The batch size to use.
        learning_rate : float (default)
            Learning rate to use in optimisation using Adam optimiser.
        epochs : int (default=2)
            Number of epochs to optimise for.
        L1_output_lambda : float (default=10e-5)
            The weighting applied to L1 regularisation on the output of the
            convolution - enforces sparsity in the result.
        L1_weight_lambda : float (default=0)
            The weighting applied to L1 regularisation to `W` - enforces
            sparsity in the convolutional filters.
        """
        self.BATCH = BATCH
        self.S_act = activation_L1
        self.S_w = weight_L1

        self.BUFFER_SIZE = images.shape[0]
        self.in_shape = images.shape

        # Batch and shuffle the data
        train_dataset = tf.data.Dataset.from_tensor_slices(
            images.astype('float32')).shuffle(self.BUFFER_SIZE).batch(BATCH)
        # optimizer
        optimizer = tf.optimizers.Adam( learning_rate=learning_rate)
        losses = self.train(train_dataset, epochs, optimizer)
        return losses

    def predict(self,
                images : np.ndarray) -> np.ndarray:
        """Forward pass of the convolution.

        Calculates the forward pass of the convolution, i.e. images convolved
        with the filter.

        Parameters
        ----------
        images : numpy.ndarray
            The input images.
        
        Returns
        -------
        out_images : numpy.ndarray
            The result from the convolution.
        """
        Nim = images.shape[0]
        aux = tf.nn.conv2d(images[0:self.BATCH,:,:,:],
                           self.W,self.stride_size, padding='SAME')

        out_images = np.zeros((Nim,aux.shape[1],aux.shape[2],aux.shape[3]),
                              dtype=np.float32)
            
        for ii in range(0,Nim,self.BATCH):
            out_images[ii:ii+self.BATCH,:,:,:] = tf.nn.conv2d(
                images[ii:ii+self.BATCH,:,:,:],
                self.W,
                self.stride_size,
                padding='SAME')
        return out_images

    def inverse(self,
                encoded : np.ndarray) -> np.ndarray:
        """Performs inverse convolution using the transpose.

        Performs a transpose convolution with the filter `W` to get the inverse
        transformation.

        Parameters
        ----------
        encoded : numpy.ndarray
            Encoded array to be inverted.

        Returns
        -------
        reconstructed : numpy.ndarray
            The reconstructed image from the inverse transformation.
        """
        Nim = encoded.shape[0]
        reconstructed = np.zeros((Nim,*self.in_shape[1:]))
        for ii in range(0,Nim,self.BATCH):
            # DECODER
            reconstructed[ii:ii+self.BATCH,:,:,:] = tf.nn.conv2d_transpose(
                encoded[ii:ii+self.BATCH,:,:,:],
                self.W,
                (self.BATCH,*self.in_shape[1:]),
                self.stride_size, padding='SAME')

        return reconstructed

    @tf.function()  # Precompile
    def train_step(self,
                   inputs : np.ndarray,
                   optimizer : tf.keras.optimizers.Optimizer
                   ) -> List[float]:
        """One training step.

        Performs one step of training and back propogation.

        Parameters
        ----------
        inputs : numpy.ndarray
            Batch to train one.
        optimizer : tensorflow.keras.optimiser_v2.OptimizerV2
            The optimising to use.
        
        Returns
        -------
        current_loss : float
            The total loss for this batch.
        L2_loss : float
            The L2 reconstruction loss for this batch.
        """
        with tf.GradientTape() as tape:
            encoded = tf.nn.conv2d(
                inputs,
                self.W,
                self.stride_size,
                padding='SAME')
            decoded = tf.nn.conv2d_transpose(
                encoded,
                self.W,
                (inputs.shape[0],*self.in_shape[1:]),
                self.stride_size,
                padding='SAME')

            L2_loss = self.loss_MSE(decoded, inputs)            
            current_loss = L2_loss + self.S_act*self.loss_MAE(encoded) + \
                self.S_w*self.loss_MAE(self.W)    

        grads = tape.gradient(current_loss , self.W)
        optimizer.apply_gradients([(grads , self.W)])
        
        return current_loss, L2_loss

    def train(self, dataset, epochs, optimizer):
        """Training loop.

        The training loop for certain number of epochs. Prints out the losses
        after each training epoch.

        Parameters
        ----------
        dataset : tensorflow.data.Dataset
            The dataset to optimise over.
        epochs : int
            Number of epochs.
        optimizer : tensorflow.keras.optimiser_v2.OptimizerV2
            Optimiser to use.
        """
        losses = []
        for epoch in range(epochs):
            for image_batch in dataset:
                summ_loss, L2_loss = self.train_step(image_batch, optimizer)
                losses.append([summ_loss, L2_loss])
            tf.summary.scalar('loss', data=summ_loss, step=epoch)
            print ('Total loss {}, L2 loss {}'.format(summ_loss, L2_loss))
        return losses

    def exact_inverse(self, input_shape, encoded):
        """Computes exact inverse.

        Uses the double block toeplitz matrix that representations the
        convolution transform in order to compute an exact inverse.

        Parameters
        ----------
        input_shape : numpy.ndarray
            The shape of input used to generated `encoded`.
        encoded : numpy.ndarray
            The data to be inverted using the inverse transformation.
        
        Returns
        -------
        exact_inv : numpy.ndarray
            Exact inverse of `encoded`.
        """
        transpose_order = [0, 3, 1, 2]
        # (B, C, H, W)
        shape_T = [input_shape[i] for i in transpose_order]
        # (B, C, H, W)
        encoded_T = tf.transpose(encoded, perm=transpose_order)
        # (C*H*W, C*H*W)
        T = tf.sparse.to_dense(self.sparse_toeplitz(shape_T))
        inv_T = tf.transpose(tf.linalg.inv(T))
         # (B, C*H*W)
        exact_inv_T = tf.reshape(encoded_T, (encoded.shape[0], -1)) @ inv_T
        # (B, C, H, W)
        exact_inv = tf.reshape(exact_inv_T, shape_T)
        # (B, H, W, C)
        exact_inv = tf.transpose(exact_inv, perm=[0, 2, 3, 1])
        return exact_inv

    def sparse_toeplitz(self, input_shape):
        '''Generates convolutional matrix
        Calculates the toeplitz matrix corresponding to the convolution using
        sparse tensors.
        Parameters
        ----------
        input_shape : List[int]
            The shape of the input.
        
        Returns
        -------
        T_sparse : tensorflow.sparse.SparseTensor
            The toeplitz matrix as a sparse matrix.
        '''
        W_T = tf.transpose(self.W, perm=[2, 3, 0, 1]).numpy() # (C_o, C_i, h, w)
        if self.T_idxs is None or self.f_idxs is None:
            self.T_idxs, self.f_idxs = utils.get_toeplitz_idxs(
                    W_T.shape, input_shape[1:],
                    (self.stride_size, self.stride_size))
        T_sparse = utils.get_sparse_toeplitz(W_T, input_shape[1:],
                                        self.T_idxs, self.f_idxs)
        return tf.sparse.reorder(T_sparse)  

    def logdetJ(self, input_shape):
        """Calculates log det(J)
        Calculates the logarithm of the determinant of the Jacobian of the
        transformation.
        """
        T_sparse = self.sparse_toeplitz(input_shape)
        logabsdet = tf.linalg.slogdet(tf.sparse.to_dense(T_sparse))[1]
        logsdetJ = tf.repeat(logabsdet, input_shape[0])
        return logsdetJ


class InvertibleConv2dEmerging(tf.keras.layers.Layer):
    """
    """
    def __init__(self, input_shape, ksize=3, dilation=1):
        assert (ksize - 1) % 2 == 0
        super(InvertibleConv2dEmerging, self).__init__()
        self.kcent = (ksize - 1) // 2
        self.input_channels = input_shape[-1]
        self.height, self.width = input_shape[0], input_shape[1]
        mask_np = utils.get_conv_square_ar_mask(
            ksize, ksize, self.input_channels, self.input_channels,
            zerodiagonal=False)
        mask_upsidedown_np = mask_np[::-1, ::-1, ::-1, ::-1].copy()
        self.mask = tf.constant(mask_np)
        self.mask_upsidedown = tf.constant(mask_upsidedown_np)
        self.dilation = dilation
        filter_shape = [ksize, ksize, self.input_channels, self.input_channels]
        w1_np = utils.get_conv_weight_np(filter_shape)
        w2_np = utils.get_conv_weight_np(filter_shape)
        w1_np = w1_np * mask_np
        w2_np = w2_np * mask_upsidedown_np
        self.w1 = tf.Variable(w1_np, dtype=tf.float32, trainable=True)
        self.w2 = tf.Variable(w2_np, dtype=tf.float32, trainable=True)
        self.b = tf.Variable(
            tf.zeros_initializer()(shape=[1, 1, 1, self.input_channels]),
            trainable=True)

        self.loss_MSE = tf.keras.losses.MeanSquaredError()
        self.loss_MAE = tf.keras.regularizers.L1()
    
    def predict(self, z):
        z = tf.nn.conv2d(
            z, self.w1, [1, 1, 1, 1],
        dilations=[1, self.dilation, self.dilation, 1],
        padding='SAME', data_format='NHWC')
        
        z = tf.nn.conv2d(
            z, self.w2, [1, 1, 1, 1],
        dilations=[1, self.dilation, self.dilation, 1],
        padding='SAME', data_format='NHWC')
        
        z = z + self.b
        return z
    
    def inverse(self, z):
        x = tf.py_function(
            Inverse(is_upper=1, dilation=self.dilation),
            inp=[z, self.w2, self.b],
            Tout=tf.float32,
            name='conv2dinverse2')

        x = tf.py_function(
            Inverse(is_upper=0, dilation=self.dilation),
            inp=[x, self.w1, tf.zeros_like(self.b)],
            Tout=tf.float32,
            name='cov2dinverse1')
        return x
    
    def fit(self,
            images : np.ndarray,
            BATCH : int = 16,
            learning_rate : float = 0.001,
            epochs : int = 2,
            activation_L1 : float = 0) -> None:
        """Trains to model for reconstruction.

        Optimises the model using Adam optimiser for the L2 loss between the
        input and the reconstructed input (one convolution then transposed
        convolution).

        Parameters
        ----------
        images : numpy.ndarray
            The images to train the layer on. Must be of shape `[N, H, W, C]`.
        batch : int (default=16)
            The batch size to use.
        learning_rate : float (default)
            Learning rate to use in optimisation using Adam optimiser.
        epochs : int (default=2)
            Number of epochs to optimise for.
        L1_output_lambda : float (default=10e-5)
            The weighting applied to L1 regularisation on the output of the
            convolution - enforces sparsity in the result.
        L1_weight_lambda : float (default=0)
            The weighting applied to L1 regularisation to `W` - enforces
            sparsity in the convolutional filters.
        """
        self.BATCH = BATCH
        self.S_act = activation_L1

        self.BUFFER_SIZE = images.shape[0]
        self.in_shape = images.shape

        # Batch and shuffle the data
        train_dataset = tf.data.Dataset.from_tensor_slices(
            images.astype('float32')).shuffle(self.BUFFER_SIZE).batch(BATCH)
        # optimizer
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        losses = self.train(train_dataset, epochs, optimizer)
        return losses

    def train_step(self,
                   inputs,
                   optimizer):
        with tf.GradientTape() as tape:
            z = self.predict(inputs)
            recon = self.inverse(z)
            L2_loss = self.loss_MSE(recon, inputs)
            current_loss = self.S_act*self.loss_MAE(z)
            inputs_flat = tf.reshape(inputs, (inputs.shape[0], -1))
            inputs_dir = inputs_flat / tf.norm(inputs_flat, axis=1, keepdims=(1))
            z_flat = tf.reshape(z, (z.shape[0], -1))
            z_dir = z_flat / tf.norm(z_flat, axis=1, keepdims=(1))
            current_loss = tf.reduce_sum(tf.math.abs(tf.keras.backend.batch_dot(inputs_dir, z_dir)))
        grads = tape.gradient(current_loss, self.trainable_variables)
        grads[0] = grads[0] * self.mask
        grads[1] = grads[1] * self.mask_upsidedown
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return current_loss, L2_loss

    def train(self, dataset, epochs, optimizer):
        """Training Loop
        """
        losses = []
        for epoch in range(epochs):
            for image_batch in dataset:
                summ_loss, L2_loss = self.train_step(image_batch, optimizer)
                losses.append([summ_loss, L2_loss])
            tf.summary.scalar('loss', data=summ_loss, step=epoch)
            print ('Total loss {}, L2 loss {}'.format(summ_loss, L2_loss))
        return losses

    def logdetJ(self, input_shape):
        """Calculates log det(J)
        Calculates the logarithm of the determinant of the Jacobian of the
        transformation.
        """
        log_abs_diagonal_w1 = tf.math.log(
            tf.math.abs(tf.linalg.diag_part(self.w1[self.kcent, self.kcent])))
        log_abs_diagonal_w2 = tf.math.log(
            tf.math.abs(tf.linalg.diag_part(self.w2[self.kcent, self.kcent])))
        logdet = tf.reduce_sum(log_abs_diagonal_w1)*(self.height*self.width)
        logdet += tf.reduce_sum(log_abs_diagonal_w2)*(self.height*self.width)
        logsdetJ = tf.repeat(logdet, input_shape[0])
        return logsdetJ
