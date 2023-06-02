# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:55:16 2020

@author: Samuel A. Maloney

Classes
-------
Kernel : metaclass=ABCMeta
    Abstract base class for weight function implementations. All functions are
    defined to be zero-valued for :math:`r \geq 1`.
QuadraticSpline : Kernel
    .. math::
        w(r)=
        \\begin{cases}
        -2r^2 + 1, & r < 0.5 \\\\
        2r^2 - 4r + 2, & 0.5 \\leq r \\leq 1
        \\end{cases}
SimpleCubicSpline : Kernel
    .. math:: w(r) = 2r^3 - 3r^2 + 1, \\quad r < 1
CubicSpline : Kernel
    .. math::
        w(r)=
        \\begin{cases}
        6r^3 - 6r^2 + 1, & r < 0.5 \\\\
        -2r^3 + 6r^2 - 6r + 2, & 0.5 \\leq r < 1
        \\end{cases}
QuarticSpline : Kernel
    .. math:: w(r) = -3r^4 + 8r^3 - 6r^2 + 1
QuinticSpline : Kernel
    .. math::
        w(r)=
        \\begin{cases}
        -\\frac{405}{11}r^5 + \\frac{405}{11}r^4 - \\frac{90}{11}r^2 + 1, & r < \\frac{1}{3} \\\\
        \\frac{405}{22}r^5 - \\frac{1215}{22}r^4 + \\frac{675}{11}r^3 - \\frac{315}{11}r^2 + \\frac{75}{22}r^1 + \\frac{17}{22}, & \\frac{1}{3} \\leq r < \\frac{2}{3} \\\\
        -\\frac{81}{22}r^5 + \\frac{405}{22}r^4 - \\frac{405}{11}r^3 + \\frac{405}{11}r^2 - \\frac{405}{22}r^1 + \\frac{81}{22}, & \\frac{2}{3} \\leq r < 1
        \\end{cases}
    from https://doi.org/10.1061/(ASCE)EM.1943-7889.0001176 page 21 (here normalised to 1 at r=0)
SimpleQuinticSpline : Kernel
    .. math:: w(r) = -6r^5 + 15r^4 - 10r^3 + 1
SepticSpline : Kernel
    .. math:: w(r) = -8r^7 + 35r^6 - 56r^5 + 35r^4 - 7r^2 + 1
GenericSpline : Kernel
    .. math:: w(r) = (1 - r^2)^n
Gaussian : Kernel
    .. math:: w(r) = \\frac{\\exp(-9r^2) - \\exp(-9)}{1 - \\exp(-9)}
Bump : Kernel
    .. math:: w(r) = \\exp(\\frac{r^2}{r^2 - 1})
"""

from abc import ABCMeta, abstractmethod
import numpy as np

class Kernel(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self): pass

    @abstractmethod
    def w(self, r):
        """Compute kernel weight function value.

        Parameters
        ----------
        r : numpy.ndarray, dtype='float64', shape=(n,)
            Distances from evaluation points to node point. Must be positive.

        Returns
        -------
        w : numpy.ndarray, dtype='float64', shape=(n,)
            Values of the kernel function at the given distances.

        """
        pass

    @abstractmethod
    def dw(self, r):
        """Compute kernel weight function value and its radial derivative.

        Parameters
        ----------
        r : numpy.ndarray, dtype='float64', shape=(n,)
            Distances from evaluation points to node point. Must be positive.

        Returns
        -------
        w : numpy.ndarray, dtype='float64', shape=(n,)
            Values of the kernel function at the given distances.
        dwdr : numpy.ndarray, dtype='float64', shape=(n,)
            Values of the radial derivative at the given distances.

        """
        pass

    @abstractmethod
    def d2w(self, r):
        """Compute kernel weight function and its radial derivatives.

        Parameters
        ----------
        r : numpy.ndarray, dtype='float64', shape=(n,)
            Distances from evaluation points to node point. Must be positive.

        Returns
        -------
        w : numpy.ndarray, dtype='float64', shape=(n,)
            Values of the kernel function at the given distances.
        dwdr : numpy.ndarray, dtype='float64', shape=(n,)
            Values of the radial derivative at the given distances.
        d2wdr2 : numpy.ndarray, dtype='float64', shape=(n,)
            Values of the 2nd order radial derivative at the given distances.

        """
        pass

    def __call__(self, r):
        return self.w(r)

class LinearSpline(Kernel):
    @property
    def name(self):
        return 'linear'

    def w(self, r):
        i0 = r <= 1
        w = np.zeros(r.size)
        if i0.any():
            w[i0] = 1 - r[i0]
        return w

    def dw(self, r):
        i0 = r <= 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        if i0.any():
            w[i0] = 1 - r[i0]
            dwdr[i0] = -1.
        return w, dwdr

    def d2w(self, r):
        i0 = r <= 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        d2wdr2 = w.copy()
        if i0.any():
            w[i0] = 1 - r[i0]
            dwdr[i0] = -1.
            d2wdr2[i0] = 0.
        return w, dwdr, d2wdr2

class QuadraticSpline(Kernel):
    @property
    def name(self):
        return 'quadratic'

    def w(self, r):
        i0 = r < 0.5
        i1 = np.logical_xor(r <= 1, i0)
        w = np.zeros(r.size)
        if i0.any():
            r1 = r[i0]
            w[i0] = -2.*r1*r1 + 1.
        if i1.any():
            r1 = r[i1]
            w[i1] = 2.*r1*r1 - 4.*r1 + 2.
        return w

    def dw(self, r):
        i0 = r < 0.5
        i1 = np.logical_xor(r <= 1, i0)
        w = np.zeros(r.size)
        dwdr = w.copy()
        if i0.any():
            r1 = r[i0]
            w[i0] = -2.*r1*r1 + 1.
            dwdr[i0] = -4.*r1
        if i1.any():
            r1 = r[i1]
            w[i1] = 2.*r1*r1 - 4.*r1 + 2.
            dwdr[i1] = 4.*r1 - 4.
        return w, dwdr

    def d2w(self, r):
        i0 = r < 0.5
        i1 = np.logical_xor(r <= 1, i0)
        w = np.zeros(r.size)
        dwdr = w.copy()
        d2wdr2 = w.copy()
        if i0.any():
            r1 = r[i0]
            w[i0] = -2.*r1*r1 + 1.
            dwdr[i0] = -4.*r1
            d2wdr2[i0] = -4.
        if i1.any():
            r1 = r[i1]
            w[i1] = 2.*r1*r1 - 4.*r1 + 2.
            dwdr[i1] = 4.*r1 - 4.
            d2wdr2[i1] = 4.
        return w, dwdr, d2wdr2

class SimpleCubicSpline(Kernel):
    @property
    def name(self):
        return 'simpleCubic'

    def w(self, r):
        i0 = r <= 1
        w = np.zeros(r.size)
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1
            w[i0] = 2*r3 - 3*r2 + 1
        return w

    def dw(self, r):
        i0 = r <= 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1
            w[i0] = 2*r3 - 3*r2 + 1
            dwdr[i0] = 6*r2 - 6*r1
        return w, dwdr

    def d2w(self, r):
        i0 = r <= 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        d2wdr2 = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1
            w[i0] = 2*r3 - 3*r2 + 1
            dwdr[i0] = 6*r2 - 6*r1
            d2wdr2[i0] = 12*r1 - 6
        return w, dwdr, d2wdr2

class CubicSpline(Kernel):
    @property
    def name(self):
        return 'cubic'

    def w(self, r):
        i0 = r < 0.5
        i1 = np.logical_xor(r < 1, i0)
        w = np.zeros(r.size)
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1
            w[i0] = 6*r3 - 6*r2 + 1
        if i1.any():
            r1 = r[i1]; r2 = r1*r1; r3 = r2*r1
            w[i1] = -2*r3 + 6*r2 - 6*r1 + 2
        return w

    def dw(self, r):
        i0 = r < 0.5
        i1 = np.logical_xor(r < 1, i0)
        w = np.zeros(r.size)
        dwdr = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1
            w[i0] = 6*r3 - 6*r2 + 1
            dwdr[i0] = 18*r2 - 12*r1
        if i1.any():
            r1 = r[i1]; r2 = r1*r1; r3 = r2*r1
            w[i1] = -2*r3 + 6*r2 - 6*r1 + 2
            dwdr[i1] = -6*r2 + 12*r1 - 6
        return w, dwdr

    def d2w(self, r):
        i0 = r < 0.5
        i1 = np.logical_xor(r < 1, i0)
        w = np.zeros(r.size)
        dwdr = w.copy()
        d2wdr2 = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1
            w[i0] = 6*r3 - 6*r2 + 1
            dwdr[i0] = 18*r2 - 12*r1
            d2wdr2[i0] = 36*r1 - 12
        if i1.any():
            r1 = r[i1]; r2 = r1*r1; r3 = r2*r1
            w[i1] = -2*r3 + 6*r2 - 6*r1 + 2
            dwdr[i1] = -6*r2 + 12*r1 - 6
            d2wdr2[i1] = -12*r1 + 12
        return w, dwdr, d2wdr2

class QuarticSpline(Kernel):
    @property
    def name(self):
        return 'quartic'

    def w(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2
            w[i0] = -3*r4 + 8*r3 - 6*r2 + 1
        return w

    def dw(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2
            w[i0] = -3*r4 + 8*r3 - 6*r2 + 1
            dwdr[i0] =  -12*r3 + 24*r2 - 12*r1
        return w, dwdr

    def d2w(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        d2wdr2 = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2
            w[i0] = -3*r4 + 8*r3 - 6*r2 + 1
            dwdr[i0] =  -12*r3 + 24*r2 - 12*r1
            d2wdr2[i0] = -36*r2 + 48*r1 - 12
        return w, dwdr, d2wdr2

class QuinticSpline(Kernel):
# https://doi.org/10.1061/(ASCE)EM.1943-7889.0001176 page 21
    @property
    def name(self):
        return 'quintic'

    def w(self, r):
        i0 = r < 1/3
        i1 = np.logical_xor(r < 2/3, i0)
        i2 = np.logical_xor(r < 1, i0 + i1)
        w = np.zeros(r.size)
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i0] = -405/11*r5 + 405/11*r4 - 90/11*r2 + 1
        if i1.any():
            r1 = r[i1]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i1] = 405/22*r5 - 1215/22*r4 + 675/11*r3 - 315/11*r2 + 75/22*r1 + 17/22
        if i2.any():
            r1 = r[i2]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i2] = -81/22*r5 + 405/22*r4 - 405/11*r3 + 405/11*r2 - 405/22*r1 + 81/22
        return w

    def dw(self, r):
        i0 = r < 1/3
        i1 = np.logical_xor(r < 2/3, i0)
        i2 = np.logical_xor(r < 1, i0 + i1)
        w = np.zeros(r.size)
        dwdr = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i0] = -405/11*r5 + 405/11*r4 - 90/11*r2 + 1
            dwdr[i0] = -2025/11*r4 + 1620/11*r3 - 180/11*r1
        if i1.any():
            r1 = r[i1]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i1] = 405/22*r5 - 1215/22*r4 + 675/11*r3 - 315/11*r2 + 75/22*r1 + 17/22
            dwdr[i1] = 2025/22*r4 - 2430/11*r3 + 2025/11*r2 - 630/11*r1 + 75/22
        if i2.any():
            r1 = r[i2]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i2] = -81/22*r5 + 405/22*r4 - 405/11*r3 + 405/11*r2 - 405/22*r1 + 81/22
            dwdr[i2] = -405/22*r4 + 810/11*r3 - 1215/11*r2 + 810/11*r1 - 405/22
        return w, dwdr

    def d2w(self, r):
        i0 = r < 1/3
        i1 = np.logical_xor(r < 2/3, i0)
        i2 = np.logical_xor(r < 1, i0 + i1)
        w = np.zeros(r.size)
        dwdr = w.copy()
        d2wdr2 = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i0] = -405/11*r5 + 405/11*r4 - 90/11*r2 + 1
            dwdr[i0] = -2025/11*r4 + 1620/11*r3 - 180/11*r1
            d2wdr2[i0] = -8100/11*r3 + 4860/11*r2 - 180/11
        if i1.any():
            r1 = r[i1]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i1] = 405/22*r5 - 1215/22*r4 + 675/11*r3 - 315/11*r2 + 75/22*r1 + 17/22
            dwdr[i1] = 2025/22*r4 - 2430/11*r3 + 2025/11*r2 - 630/11*r1 + 75/22
            d2wdr2[i1] = 4050/11*r3 - 7290/11*r2 + 4050/11*r1 - 630/11
        if i2.any():
            r1 = r[i2]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i2] = -81/22*r5 + 405/22*r4 - 405/11*r3 + 405/11*r2 - 405/22*r1 + 81/22
            dwdr[i2] = -405/22*r4 + 810/11*r3 - 1215/11*r2 + 810/11*r1 - 405/22
            d2wdr2[i2] = -810/11*r3 + 2430/11*r2 - 2430/11*r1 + 810/11
        return w, dwdr, d2wdr2

class SimpleQuinticSpline(Kernel):
    @property
    def name(self):
        return 'simpleQuintic'

    def w(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i0] = -8/3*r5 + 5*r4 - 10/3*r2 + 1
        return w

    def dw(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i0] = -8/3*r5 + 5*r4 - 10/3*r2 + 1
            dwdr[i0] = -40/3*r4 + 20*r3 - 20/3*r1
        return w, dwdr

    def d2w(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        d2wdr2 = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i0] = -8/3*r5 + 5*r4 - 10/3*r2 + 1
            dwdr[i0] = -40/3*r4 + 20*r3 - 20/3*r1
            d2wdr2[i0] = -160/3*r3 + 60*r2 - 20/3
        return w, dwdr, d2wdr2
    
class SepticSpline(Kernel):
    @property
    def name(self):
        return 'septicQuintic'

    def w(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            r6 = r3*r3; r7 = r4*r3
            w[i0] = -8*r7 + 35*r6 - 56*r5 + 35*r4 - 7*r2 + 1
        return w

    def dw(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            r6 = r3*r3; r7 = r4*r3
            w[i0] = -8*r7 + 35*r6 - 56*r5 + 35*r4 - 7*r2 + 1
            dwdr[i0] = -56*r6 + 210*r5 - 280*r4 + 140*r3 - 14*r1
        return w, dwdr

    def d2w(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        d2wdr2 = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            r6 = r3*r3; r7 = r4*r3
            w[i0] = -8*r7 + 35*r6 - 56*r5 + 35*r4 - 7*r2 + 1
            dwdr[i0] = -56*r6 + 210*r5 - 280*r4 + 140*r3 - 14*r1
            d2wdr2[i0] = -336*r5 + 1050*r4 - 1120*r3 + 420*r2 - 14
        return w, dwdr, d2wdr2

class GenericSpline(Kernel):
    @property
    def name(self):
        return 'generic'

    def __init__(self, n=1):
        self.n = n

    def w(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        if i0.any():
            w[i0] = (1 - r[i0]**2)**self.n
        return w

    def dw(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1;
            w[i0] = (1 - r2)**self.n
            dwdr[i0] = -2*self.n*r1*(1 - r2)**(self.n-1)
        return w, dwdr

    def d2w(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        d2wdr2 = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1;
            w[i0] = (1 - r2)**self.n
            dwdr[i0] = -2*self.n*r1*(1 - r2)**(self.n-1)
            d2wdr2[i0] = 2*self.n*((2*self.n-1)*r2 - 1)*(1 - r2)**2
        return w, dwdr, d2wdr2

class Gaussian(Kernel):
    c1 = np.exp(-9)
    c2 = 1/(1 - np.exp(-9))

    @property
    def name(self):
        return 'gaussian'

    def w(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        if i0.any():
            r2 = r[i0]**2
            w[i0] = (np.exp(-9*r2) - self.c1) / (1 - self.c1)
        return w

    def dw(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1**2
            w[i0] = (np.exp(-9*r2) - self.c1) * self.c2
            dwdr[i0] = -18*r1*np.exp(-9*r2) * self.c2
        return w, dwdr

    def d2w(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        d2wdr2 = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1**2
            w[i0] = (np.exp(-9*r2) - self.c1) * self.c2
            dwdr[i0] = -18*r1*np.exp(-9*r2) * self.c2
            d2wdr2[i0] = 18*np.exp(-9*r2)*(18*r2-1) * self.c2
        return w, dwdr, d2wdr2

class Bump(Kernel):

    @property
    def name(self):
        return 'bump'

    def w(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        if i0.any():
            r2 = r[i0]**2
            w[i0] = np.exp(r2/(r2 - 1))
        return w

    def dw(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1**2
            tmp = 1/(r2 - 1)
            w[i0] = np.exp(tmp + 1)
            dwdr[i0] = w[i0]*(-2*r1)*tmp**2
        return w, dwdr

    def d2w(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        d2wdr2 = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1**2
            tmp1 = 1/(r2 - 1)
            w[i0] = np.exp(tmp1 + 1)
            tmp2 = tmp1**2
            dwdr[i0] = w[i0]*(-2*r1)*tmp2
            d2wdr2[i0] = w[i0]*(6*r2**2 - 2)*tmp2**2
        return w, dwdr, d2wdr2