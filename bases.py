# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:16:57 2022

@author: Samuel A. Maloney

"""

import numpy as np
from abc import ABCMeta, abstractmethod

class Basis(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self): raise NotImplementedError

    @abstractmethod
    def p0(self):
        """The values of the basis function computed at the origin.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray, dtype='float64', shape=(self.size,)
            Values of the basis functions computed at the origin.

        """
        raise NotImplementedError
    
    @abstractmethod
    def dp0(self):
        """The values of the basis function derivatives computed at the origin.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray, dtype='float64', shape=(ndim, self.size)
            Values of the basis function derivatives computed at the origin.
            The rows correspond to different spatial dimensions.

        """
        raise NotImplementedError

    @abstractmethod
    def p(self, p):
        """Compute the basis function values at a given set of points.

        Parameters
        ----------
        p : numpy.ndarray, dtype='float64', shape=(n, ndim)
            Coordinates of evaluation points.

        Returns
        -------
        numpy.ndarray, dtype='float64', shape=(n, self.size)
            Values of basis functions at given points.

        """
        raise NotImplementedError

    @abstractmethod
    def dp(self, p):
        """Compute the basis function derivatives at a given point.

        Parameters
        ----------
        p : numpy.ndarray, dtype='float64', shape=(ndim,)
            Coordinates of evaluation point.

        Returns
        -------
        numpy.ndarray, dtype='float64', shape=(ndim, self.size)
            Derivatives of basis functions at the given point. The rows
            correspond to different spatial dimensions.

        """
        raise NotImplementedError

    @abstractmethod
    def d2p(self, p):
        """Compute the basis function 2nd derivatives at a given point.

        Parameters
        ----------
        p : numpy.ndarray, dtype='float64', shape=(ndim,)
            Coordinates of evaluation point.

        Returns
        -------
        numpy.ndarray, dtype='float64', shape=(ndim, self.size)
            2nd derivatives of basis functions at the given point. The rows
            correspond to different spatial dimensions.

        """
        raise NotImplementedError

    def __call__(self, p):
        return self.p(p)


class LinearBasis(Basis):
    @property
    def name(self):
        return 'linear'

    def __init__(self, ndim):
        self.ndim = ndim
        self.size = ndim + 1
        self._p0 = np.concatenate(([1.], np.zeros(ndim)))
        self._dp = np.hstack((np.zeros((ndim,1)), np.eye(ndim)))
        self._d2p = np.zeros((ndim, ndim+1))

    def p0(self):
        return self._p0.copy()

    def dp0(self):
        return self._dp.copy()

    def p(self, p):
        nPoints = p.size // self.ndim
        return np.hstack((np.ones((nPoints, 1)), p.reshape(-1, self.ndim)))

    def dp(self, p):
        if p.size == self.ndim:
            return self._dp.copy()
        else:
            nPoints = p.size // self.ndim
            return np.repeat(self._dp[np.newaxis,:,:], nPoints, axis=0)

    def d2p(self, p=None):
        return self._d2p.copy()


class QuadraticBasis(Basis):
    @property
    def name(self):
        return 'quadratic'

    def __init__(self, ndim):
        self.ndim = ndim
        self.size = (ndim+1)*(ndim+2) // 2
        self._p0 = np.concatenate(([1.], np.zeros(self.size - 1)))
        self._dpLinear = np.hstack((np.zeros((ndim,1)), np.eye(ndim)))
        self._dp0 = np.hstack((self._dpLinear,
                               np.zeros((ndim, ndim*(ndim+1)//2))))
        if self.ndim == 1:
            self._d2p = np.array((0., 0., 2.))
        elif self.ndim == 2:
            self._d2p = np.array(((0., 0., 0., 2., 0., 0.),
                                  (0., 0., 0., 0., 0., 2.)))
        elif self.ndim == 3:
            self._d2p = np.array(((0., 0., 0., 0., 2., 0., 0., 0., 0., 0.),
                                  (0., 0., 0., 0., 0., 0., 0., 2., 0., 0.),
                                  (0., 0., 0., 0., 0., 0., 0., 0., 0., 2.)))

    def p0(self):
        return self._p0.copy()

    def dp0(self):
        return self._dp0.copy()

    def p(self, p):
        nPoints = p.size // self.ndim
        if self.ndim == 1:
            return np.hstack((np.ones((nPoints,1)), p.reshape(-1,1),
                              p.reshape(-1,1)**2))
        elif self.ndim == 2:
            x = p.reshape(-1,2)[:,0:1]
            y = p.reshape(-1,2)[:,1:2]
            return np.hstack((np.ones((nPoints,1)), p.reshape(-1,2),
                              x**2, x*y, y**2))
        elif self.ndim == 3:
            x = p.reshape(-1,3)[:,0:1]
            y = p.reshape(-1,3)[:,1:2]
            z = p.reshape(-1,3)[:,2:3]
            return np.hstack((np.ones((nPoints,1)), p.reshape(-1,3),
                              x**2, x*y, x*z, y**2, y*z, z**2))

    def dp(self, p):
        if p.size == self.ndim:
            if self.ndim == 1:
                return np.hstack((self._dpLinear, 2.*p.reshape(1,1)))
            elif self.ndim == 2:
                x = p.flat[0]
                y = p.flat[1]
                return np.hstack((self._dpLinear, ((2.*x, y,  0. ),
                                                   ( 0. , x, 2.*y))))
            elif self.ndim == 3:
                x = p.flat[0]
                y = p.flat[1]
                z = p.flat[2]
                return np.hstack((
                    self._dpLinear, ((2.*x, y , z ,  0. , 0.,  0. ),
                                     ( 0. , x , 0., 2.*y, z ,  0. ),
                                     ( 0. , 0., x ,  0. , y , 2.*z)) ))
        else:
            nPoints = p.size // self.ndim
            dp = np.zeros((nPoints, self.ndim, self.size))
            dp[:,:,:self.ndim+1] = \
                np.repeat(self._dpLinear[np.newaxis,:,:], nPoints, axis=0)
            if self.ndim == 1:
                dp[:,:,-1] = 2.*p.reshape(1,1)
            elif self.ndim == 2:
                x = p.reshape(-1,2)[:,0]
                y = p.reshape(-1,2)[:,1]
                dp[:,0,-3] = 2.*x
                dp[:,0,-2] = y
                dp[:,1,-2] = x
                dp[:,1,-1] = 2.*y
            elif self.ndim == 3:
                x = p.reshape(-1,3)[:,0]
                y = p.reshape(-1,3)[:,1]
                z = p.reshape(-1,3)[:,2]
                dp[:,0,-6] = 2.*x
                dp[:,0,-5] = y
                dp[:,0,-4] = z
                dp[:,1,-5] = x
                dp[:,1,-3] = 2.*y
                dp[:,1,-2] = z
                dp[:,2,-4] = x
                dp[:,2,-2] = y
                dp[:,2,-1] = 2.*z
            return dp

    def d2p(self, p=None):
        return self._d2p.copy()
