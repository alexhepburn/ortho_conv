# File containing classes for Gaussianisation

from typing import Tuple, List
import numpy as np

from rbig._src.uniform import MarginalHistogramUniformization
from rbig._src.invcdf import InverseGaussCDF
from rbig._src.total_corr import information_reduction

__all__ = ['HistogramGaussianisation']


class HistogramGaussianisation:
    """A class for Gaussianisation via histograms.

    Attributes
    ----------
    transformations : List[object]
        The transformations that lead to marginal Gaussianisation. Populated
        after `fit` has been called.
    """
    def __init__(self):
        """
        Constructs a HistogramGaussianisation class.
        """
        self.transformations = []
    
    def fit(self,
            images : np.ndarray,
            n_samples : int = 10000) -> np.ndarray:
        """Fits the transformations to the data.

        Fits the marginal Gaussianisation transform to the data. 

        Parameters
        ----------
        images : numpy.ndarray
            The images to train the layer on. Must be of shape `[N, H, W, C]`.
        n_samples : int (default=10000)
            Number of samples to fit the distribution to.
        
        Returns
        -------
        Z_G : numpy.ndarray
            The marginally Gaussianised data.
        """
        aux = np.reshape(images,(np.prod(images.shape[0:-1]),images.shape[3]))
        self.Z = np.random.permutation(aux)[0:n_samples,:]

        Z_G, self.transformations, self.log_pZ_1, self.MI_1 = \
            self.marginal_gaussianization(self.Z)

        self.im_shape = images.shape

        return Z_G

    def predict(self,
                images : np.ndarray,
                batch : int = 100):
        """Performs marginal Gaussianisation on the data.

        Marginally Gaussianises the data according to the trained
        transformations.

        Parameters
        ----------
        images : numpy.ndarray
            The images to marginally Gaussianise.
        batch : int (default=100)
            Number of images to apply the transformation to at the same time.
        
        Returns
        -------
        images_G : numpy.ndarray
            The Gaussianised images.
        """

        aux = np.reshape(
            images,
            (images.shape[0]*np.prod(self.im_shape[1:-1]),self.im_shape[3]))
        
        Nim = aux.shape[0]
        
        BATCH_loop = self.im_shape[1]*self.im_shape[2]*batch
        images_G = np.zeros(images.shape, dtype=np.float32)

        cada = 0
        for ii in range(0,Nim,BATCH_loop):
            Z_U2 = self.transformations[0].forward(aux[ii:ii+BATCH_loop,:])
            Z_G2 = self.transformations[1].forward(Z_U2)
            images_G[cada:cada+batch,:,:,:] = np.reshape(
                Z_G2,(batch,*images.shape[1:]))    
            cada = cada+batch

        return images_G

    def inverse(self,
                images : np.ndarray,
                batch : int = 100):
        """Applies inverse transform.

        Applies the inverse transform according to the learned transformations.

        Parameters
        ----------
        images : numpy.ndarray
            The images to apply the inverse transform to.
        batch : int (default=100)
            Number of images to apply the transformation to at the same time.
        
        Returns
        -------
        images_I : numpy.ndarray
            The images with the inverse transform applied.
        """
        aux = np.reshape(
            images,
            (images.shape[0]*np.prod(self.im_shape[1:-1]),self.im_shape[3]))

        # To apply
        Nim = aux.shape[0]
        
        BATCH_loop = self.im_shape[1]*self.im_shape[2]*batch
        images_I = np.zeros(images.shape, dtype=np.float32)

        cada = 0
        for ii in range(0,Nim,BATCH_loop):
            Z_Ui = self.transformations[1].inverse(aux[ii:ii+BATCH_loop,:])
            Z_i = self.transformations[0].inverse(Z_Ui)
            images_I[cada:cada+batch,:,:,:] = np.reshape(
                Z_i,(batch,*images.shape[1:]))    
            cada = cada+batch

        return images_I
    
    def marginal_gaussianization(self,
                                 Z : np.ndarray
        ) -> Tuple[np.ndarray, List[object], np.ndarray, float]:
        """Gets the marginal Gaussianisation transformations.

        Gets the marginal Gaussianisation transformations using the functions
        from the RBIG package https://github.com/IPL-UV/rbig/

        Parameters
        ----------
        Z : numpy.ndarray
            The data to marginally Gaussianise.
        
        Returns
        -------
        Z_G : numpy.ndarray
            Marginally Gaussianised data.
        transformations : List[object]
            List of transformations to achieve marginal Gaussianisation.
        log_pZ : numpy.ndarray
            The log probability of the data.
        MI : float
            The mutual information.
        """
        transformations = []

        bins = "auto"
        alpha = 1e-10
        bound_ext = 0.3
        eps = 1e-5

        ibijector = MarginalHistogramUniformization(Z, bound_ext=bound_ext, bins=bins, alpha=alpha)
        Z_U = ibijector.forward(Z).astype(np.float32)
        log_pZ = ibijector.gradient(Z)
        transformations.append(ibijector)

        # Inverse Gauss CDF
        ibijector = InverseGaussCDF(eps=eps)
        Z_G = ibijector.forward(Z_U).astype(np.float32)
        log_pZ += ibijector.gradient(Z_U)
        transformations.append(ibijector)

        MI = information_reduction(Z,Z_G)

        return Z_G, transformations, log_pZ, MI
