# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:47:07 2020

@author: Samuel A. Maloney

"""

from scipy.special import roots_legendre
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la
import numpy as np
import integrators
import mappings
import boundaries
import kernels
import bases
import ssqr


class FciMlsSim:
    """Class for flux-coordinate independent MLS (FCIMLS) method.
    Implements the convection-diffusion equation on a rectangular domain
    (0, xmax) X (0, 1).

    Attributes
    ----------
    NX : int
        Number of planes along x-dimension. Must be NX >= 2.
    NY : int
        Number of nodes on each plane. Must be NY >= 2.
    nodeX : numpy.ndarray, shape=(NX+1,)
        x-coords of FCI planes (includes right boundary).
    dx : numpy.ndarray, shape=(NX,)
        Spacing between FCI planes
    nodeY : numpy.ndarray, shape=(NX+1, NY+1)
        y-coords of nodes on each FCI plane (includes right/top boundaries).
    idy : numpy.ndarray, shape=(NX+1, NY)
        1/spacing between nodes on each FCI plane (includes right boundary).
    nDoFs : int
        Number of unique nodal points in the simulation domain (equals NX*NY).
    mapping : mappings.Mapping
        Mapping function for the FCIMLS method.
    boundary : boundaries.Boundary
        Object defining boundary conditions and shape function support sizes.
    kernel : kernels.Kernel
        Kernel object defining the weight function and its derivatives.
    basis : bases.Basis
        Complete polynomial basis defining the MLS aproximation.
    velocity : np.array([vx,vy], dtype='float64')
        Background velocity of the fluid.
    diffusivity : {numpy.ndarray, float}
        Diffusion coefficient for the quantity of interest.
        If an array, it must have shape (ndim,ndim). If a float, it will
        be converted to diffusivity*np.eye(ndim, dtype='float64').
    f : callable
        Forcing function. Must take 2D array of points and return 1D array.
    NQX : int
        Number of quadrature cell divisions between FCI planes.
    NQY : int
        Number of quadrature cell divisions in y-direction.
    Qord : int
        Number of quadrature points in each grid cell along one dimension.
    quadType : string, optional
        Type of quadrature to be used. Must be either 'gauss' or 'uniform'.
        Produces Gauss-Legendre or Newton-Cotes points/weights respectively.
    massLumping : bool, optional
        Determines whether mass-lumping was used to calculate M matrix.
    K : scipy.sparse.csr_matrix
        The stiffness matrix from the diffusion term
    A : scipy.sparse.csr_matrix
        The advection matrix
    M : scipy.sparse.csr_matrix
        The mass matrix from the time derivative
    b : numpy.ndarray, shape=(nDoFs,)
        RHS forcing vector generated from source/sink function f.
    integrator : Integrator
        Object defining time-integration scheme to be used.

    Methods
    -------
    selectKernel(self, kernel)
        Register the 'self.kernel' object to the correct kernel.
    selectBasis(self, basis)
        Register the 'self.basis' object to the correct basis set.
    phi(self, point)
        Compute shape function value evaluated at point.
    dphi(self, point)
        Compute shape function value and gradient evaluated at point.
    d2phi(self, point)
        Compute shape function value and laplacian evaluated at point.
    setInitialConditions(self, u0, mapped=True)
        Initialize the nodal coefficients for the given IC.
    computeSpatialDiscretization(self, f=None, NQX=1, NQY=None, Qord=2,
                                 quadType='gauss', massLumping=False, **kwargs)
        Assemble the system discretization matrices K, A, M in CSR format.
    solve(self)
        Reconstruct the final solution vector, u, from the shape functions.

    """

    def __init__(self, NX, NY, mapping, boundary=('periodic', 1.5),
                 kernel='cubic', basis='linear', velocity=None, diffusivity=0.,
                 px=0., py=0., seed=None, xmax=1., ymax=1., **kwargs):
        """Initialize attributes of FCIMLS simulation class

        Parameters
        ----------
        NX : int
            Number of planes along x-dimension. Must be NX >= 2.
        NY : int
            Number of nodes on each plane. Must be NY >= 2.
        mapping : mappings.Mapping
            Mapping function for the FCIMLS method.
            Must be an object derived from fcimls.mappings.Mapping.
        boundary : {boundaries.Boundary, (string, float)}, optional
            Either a boundaries.Boundary object, or (string, float) tuple.
            String must be either 'periodic' or 'Dirichlet' and the float is
            the shape function support size relative to uniform grid spacing.
            The default is ('periodic', 1.5).
        kernel : {string, kernels.Kernel}, optional
            Function to be used for the kernel weighting function.
            If a string, must be in ['linear', 'quadratic', 'cubic',
            'simpleCubic', 'quartic', 'quintic', 'simpleQuintic', 'gaussian',
            'bump']. Otherwise a Kernel object can be directly specified.
            The default is 'cubic'.
        basis : {string, bases.Basis}, optional
            Complete polynomial basis defining the MLS aproximation.
            If a string, must be either 'linear' or 'quadratic'.
            The default is 'linear'.
        velocity : array_like, optional
            Background velocity vector of the fluid, must be of length ndim.
        diffusivity : {array_like, float}, optional
            Diffusion coefficient for the quantity of interest.
            If an array, it must have shape (ndim,ndim). If a float, it will
            be converted to diffusivity*np.eye(ndim, dtype='float').
            The default is 0.
        px : float, optional
            Max amplitude of random perturbations added to FCI plane locations.
            Size is relative to grid spacing (px*xmax/NX). The default is 0.
        py : float, optional
            Max amplitude of random perturbations added to node y-coords.
            Size is relative to grid spacing (py/NY). The default is 0.
        seed : {None, int, array_like[ints], numpy.random.SeedSequence}, optional
            A seed to initialize the RNG. If None, then fresh, unpredictable
            entropy will be pulled from the OS. The default is None.
        xmax : float, optional
            Maximum x-coordinate of the rectuangular domain. The default is 1.
        xmax : float, optional
            Maximum y-coordinate of the rectuangular domain. The default is 1.
        **kwargs
            Keyword arguments

        """
        NX = int(NX) # 'numpy.int**' classes can cause problems with SuiteSparse
        NY = int(NY)
        self.ndim = 2
        self.NX = NX
        self.NY = NY
        self.xmax = xmax
        self.ymax = ymax
        if isinstance(mapping, mappings.Mapping):
            self.mapping = mapping
        else:
            raise TypeError('mapping must be of type fcimls.mappings.Mapping')
        if velocity is None:
            self.velocity = np.zeros(self.ndim)
        else:
            self.velocity = np.array(velocity, dtype='float').reshape(-1)
        if isinstance(diffusivity, np.ndarray):
            self.diffusivity = diffusivity
        else:
            self.diffusivity = np.array(diffusivity, dtype='float')
            if self.diffusivity.shape != (self.ndim, self.ndim):
                self.diffusivity = diffusivity*np.eye(self.ndim)
        if self.diffusivity.shape != (self.ndim,self.ndim):
            raise TypeError(f"diffusivity must be (or be convertible to) a "
                f"numpy.ndarray with shape ({self.ndim}, {self.ndim}).")
        rng = np.random.Generator(np.random.PCG64(seed))
        if "nodeX" in kwargs:
            self.nodeX = kwargs["nodeX"]
        else:
            self.nodeX = xmax*np.arange(NX+1)/NX
            px *= xmax/NX
            self.nodeX[1:-1] += rng.uniform(-px, px, self.nodeX[1:-1].shape)
        self.nodeY = np.tile(np.linspace(0, 1, NY+1), NX+1).reshape(NX+1,-1)
        py /= NY
        self.nodeY[:-1,1:-1] += rng.uniform(-py, py, self.nodeY[:-1,1:-1].shape)
        self.nodeY[-1] = self.nodeY[0]
        self.dx = self.nodeX[1:] - self.nodeX[0:-1]
        self.dy = self.nodeY[:,1:] - self.nodeY[:,:-1]
        self.idy = 1. / self.dy
        if isinstance(boundary, boundaries.Boundary):
             self.boundary = boundary
        elif boundary[0].lower() in ('periodic', 'p'):
            self.boundary = boundaries.PeriodicBoundary(self, boundary[1])
        elif boundary[0].lower() in ('dirichlet', 'd'):
            self.boundary = boundaries.DirichletBoundary(self, boundary[1])
        else:
            raise TypeError(f"Unkown boundary condition: {boundary}")
        self.nDoFs = self.boundary.nDoFs
        self.nNodes = self.boundary.nNodes
        self.nodes = self.boundary.computeNodes()
        self.DoFs = self.nodes[:self.nDoFs]
        self.DoFsMapped = self.DoFs.copy()
        self.DoFsMapped[:,1] = self.boundary.mapping(self.DoFs, 0)
        self.selectKernel(kernel)
        self.selectBasis(basis)

    def selectKernel(self, kernel='cubic'):
        """Register the 'self.kernel' object to the correct kernel.

        Parameters
        ----------
        kernel : {string, kernels.Kernel}, optional
            Function to be used for the kernel weighting function.
            If a string, must be in ['linear', 'quadratic', 'cubic',
            'simpleCubic', 'quartic', 'quintic', 'simpleQuintic', 'gaussian',
            'bump']. Otherwise a Kernel object can be directly specified.
            The default is 'cubic'.

        Returns
        -------
        None.

        """
        if isinstance(kernel, kernels.Kernel):
            self.kernel = kernel.kernel
            return
        kernel = kernel.lower()
        if kernel == 'linear':
            self.kernel = kernels.LinearSpline()
        elif kernel == 'quadratic':
            self.kernel = kernels.QuadraticSpline()
        elif kernel == 'cubic':
            self.kernel = kernels.CubicSpline()
        elif kernel == 'simpleCubic':
            self.kernel = kernels.SimpleCubicSpline()
        elif kernel == 'quartic':
            self.kernel = kernels.QuarticSpline()
        elif kernel == 'quintic':
            self.kernel = kernels.QuinticSpline()
        elif kernel == 'simpleQuintic':
            self.kernel = kernels.SimpleQuinticSpline()
        elif kernel == 'gaussian':
            self.kernel = kernels.Gaussian()
        elif kernel == 'bump':
            self.kernel = kernels.Bump()
        else:
            raise TypeError(f"Unkown kernel '{kernel}'. Must be one of "
                             f"'cubic', 'quartic', or 'gaussian' or an "
                             f"obect derived from fcimls.kernels.Kernel.")

    def selectBasis(self, basis):
        """Register the 'self.basis' object to the correct basis set.

        Parameters
        ----------
        basis : {string, Basis}
            Name of the basis to be used for the shape functions, or Basis
            object. If a string, it must be either 'linear' or 'quadratic'.

        Returns
        -------
        None.

        """
        if isinstance(basis, bases.Basis):
            self.basis = basis
            return
        basis = basis.lower()
        if basis == 'linear':
            self.basis = bases.LinearBasis(self.ndim)
        elif basis == 'quadratic':
            self.basis = bases.QuadraticBasis(self.ndim)
        else:
            raise TypeError(f"Unkown basis '{basis}'. Must be either 'linear'"
                " or 'quadratic' or object derived from fcimls.bases.Basis.")

    def phi(self, point):
        """Compute shape function value at given point.
        Does not compute any derivatives.

        Parameters
        ----------
        point : numpy.ndarray, dtype='float64', shape=(ndim,)
            Coordinates of given evaluation point.

        Returns
        -------
        indices : numpy.ndarray, dtype='uint32', shape=(n,)
            Indices of nodes with non-zero support at evaluation point.
        phis : numpy.ndarray, dtype='float64', shape=(n,)
            Values of phi for all nodes in self.nodes[indices].

        """
        # ##### Centred-and-scaled MLS #####
        # # --------------------------------------
        # #     compute the moment matrix A(x)
        # # --------------------------------------
        # indices, w, displacements = self.boundary.w(point)
        # p = self.basis(displacements)
        # A = w*p.T@p
        # # --------------------------------------
        # #      compute vector c(x) and phi
        # # --------------------------------------
        # # A(x)c(x) = p(x)
        # # Backward substitution for c(x) using LU factorization for A
        # p0 = self.basis.p0()
        # lu, piv = la.lu_factor(A, overwrite_a=True, check_finite=False)
        # c = la.lu_solve((lu, piv), p0, overwrite_b=True, check_finite=False)
        # phis = c @ p.T * w
        # # np.testing.assert_allclose(phis.sum(), 1., atol=1e-10)
        # return indices, phis

        ##### Standard MLS #####
        # --------------------------------------
        #     compute the moment matrix A(x)
        # --------------------------------------
        indices, w, displacements = self.boundary.w(point)
        p = self.basis(point - displacements*self.boundary.support)
        A = w*p.T@p
        # --------------------------------------
        #      compute vector c(x) and phi
        # --------------------------------------
        # A(x)c(x) = p(x)
        # Backward substitution for c(x) using LU factorization for A
        p_x = self.basis(point)[0]
        lu, piv = la.lu_factor(A, overwrite_a=True, check_finite=False)
        c = la.lu_solve((lu, piv), p_x, overwrite_b=True, check_finite=False)
        phi = c @ p.T * w
        return indices, phi

    def dphi(self, point):
        """Compute shape function value and gradient at given point.
        Does not compute second derivatives.

        Parameters
        ----------
        point : numpy.ndarray, dtype='float64', shape=(ndim,)
            Coordinates of given evaluation point.

        Returns
        -------
        indices : numpy.ndarray, dtype='uint32', shape=(n,)
            Indices of nodes with non-zero support at evaluation point.
        phis : numpy.ndarray, dtype='float64', shape=(n,)
            Values of phi for all n nodes in self.nodes[indices].
        gradphis : numpy.ndarray, dtype='float64', shape=(n,ndim)
            Gradients of phi for all n nodes in self.nodes[indices].
            Has the form numpy.array([[dx1,dy1,dz1],[dx2,dy2,dz2]...])

        """
        # ##### Centred-and-scaled MLS (not finished) #####
        # # --------------------------------------
        # #     compute the moment matrix A(x)
        # # --------------------------------------
        # indices, w, gradw, displacements = self.boundary.dw(point)
        # p = self.basis(displacements)
        # dp = self.basis.dp(displacements)
        # A = w*p.T@p
        # dA = [gradw[:,i]*p.T@p for i in range(self.ndim)]
        # # --------------------------------------
        # #      compute matrix c
        # # --------------------------------------
        # # A(x)c(x) = p(x)
        # # A(x)c_k(x) = b_k(x)
        # # Backward substitutions, once for c(x), ndim times for c_k(x)
        # # Using LU factorization for A
        # p0 = self.basis.p0()
        # lu, piv = la.lu_factor(A, overwrite_a=True, check_finite=False)
        # c = np.empty((self.ndim + 1, self.basis.size))
        # c[0] = la.lu_solve((lu, piv), p0, overwrite_b=True, check_finite=False)
        # dp0 = self.basis.dp0()
        # for i in range(self.ndim):
        #     c[i+1] = la.lu_solve( (lu, piv), (dp0[i] - dA[i]@c[0]),
        #                           check_finite=False )
        # # --------------------------------------
        # #       compute phi and gradphi
        # # --------------------------------------
        # cp = c[0] @ p.T
        # phis = cp * w
        # gradphis = ((+c[1:]@p.T + c[0]@dp.transpose(1,2,0))*w + cp*gradw.T).T
        # # np.testing.assert_allclose(phis.sum(), 1., atol=1e-10)
        # # np.testing.assert_allclose(gradphis.sum(axis=0), (0,0), atol=1e-10)
        # return indices, phis, gradphis

        ##### Standard MLS #####
        # --------------------------------------
        #     compute the moment matrix A(x)
        # --------------------------------------
        indices, w, gradw, displacements = self.boundary.dw(point)
        p = self.basis(point - displacements*self.boundary.support)
        A = w*p.T@p
        dA = [gradw[:,i]*p.T@p for i in range(self.ndim)]
        # --------------------------------------
        #         compute matrix c
        # --------------------------------------
        # A(x)c(x)   = p(x)
        # A(x)c_k(x) = b_k(x)
        # Backward substitutions, once for c(x), ndim times for c_k(x)
        # Using LU factorization for A
        p_x = self.basis(point)[0]
        lu, piv = la.lu_factor(A, check_finite=False)
        c = np.empty((self.ndim + 1, self.basis.size))
        c[0] = la.lu_solve((lu, piv), p_x, check_finite=False)
        dp = self.basis.dp(point)
        for i in range(self.ndim):
            c[i+1] = la.lu_solve( (lu, piv), (dp[i] - dA[i]@c[0]),
                                  check_finite=False )
        # --------------------------------------
        #       compute phi and gradphi
        # --------------------------------------
        cp = c[0] @ p.T
        phis = cp * w
        gradphis = ( c[1 : self.ndim + 1]@p.T*w + cp*gradw.T).T
        # re-align gradient to global x-coordinate
        dQ = self.mapping.deriv(point)
        gradphis[:,0] = gradphis[:,0] - dQ*gradphis[:,1]
        return indices, phis, gradphis

    def d2phi(self, point):
        """Compute shape function value and laplacian at given point.
        Does not compute the 1st order gradient.

        Parameters
        ----------
        point : numpy.ndarray, dtype='float64', shape=(ndim,)
            Coordinates of given evaluation point.

        Returns
        -------
        indices : numpy.ndarray, dtype='uint32', shape=(n,)
            Indices of nodes with non-zero support at evaluation point.
        phis : numpy.ndarray, dtype='float64', shape=(n,)
            Values of phi for all n nodes in self.nodes[indices].
        grad2phis : numpy.ndarray, dtype='float64', shape=(n,ndim)
            2nd derivatives of phi for all n nodes in self.nodes[indices].
            Has the form numpy.array([[dxx1,dyy1,dzz1],[dxx2,dyy2,dzz2]...]).

        """
        raise NotImplementedError
        # --------------------------------------
        #     compute the moment matrix A(x)
        # --------------------------------------
        indices, w, gradw, grad2w, displacements = self.boundary.d2w(point)
        p = self.basis(point - displacements*self.boundary.support)
        A = w*p.T@p
        dA = [gradw[:,i]*p.T@p for i in range(self.ndim)]
        d2A = [grad2w[:,i]*p.T@p for i in range(self.ndim)]
        # --------------------------------------
        #         compute  matrix c(x)
        # --------------------------------------
        # A(x)c(x)   = p(x)
        # A(x)c_k(x) = b_k(x)
        # Backward substitutions, once for c(x), ndim times for c_k(x)
        # and ndim times for c_kk(x), using LU factorization for A
        p_x = self.basis(point)[0]
        lu, piv = la.lu_factor(A, check_finite=False)
        c = np.empty((2*self.ndim + 1, self.basis.size))
        c[0] = la.lu_solve((lu, piv), p_x, check_finite=False)
        dp = self.basis.dp(point)
        d2p = self.basis.d2p(point)
        for i in range(self.ndim):
            c[i+1] = la.lu_solve( (lu, piv), (dp[i] - dA[i]@c[0]),
                                  check_finite=False )
            c[i+1+self.ndim] = la.lu_solve( (lu, piv),
                (d2p[i] - 2.0*dA[i]@c[i+1] - d2A[i]@c[0]),
                check_finite=False )
        # --------------------------------------
        #       compute phi and gradphi
        # --------------------------------------
        cp = c[0] @ p.T
        phis = cp * w
        grad2phis = ( c[self.ndim + 1 : 2*self.ndim + 1]@p.T*w +
                      2.0*c[1 : self.ndim + 1]@p.T*gradw.T +
                      cp*grad2w.T ).T
        # TODO: re-align gradient to global x-coordinate
        return indices, phis, grad2phis

    def setInitialConditions(self, u0, mapped=True):
        """Initialize the nodal coefficients for the given IC.

        Parameters
        ----------
        u0 : {numpy.ndarray, callable}
            Initial conditions for the simulation.
            Must be an array of shape (self.nDoFs,) or a callable object
            returning such an array and taking as input the array of node
            coordinates with shape (self.nDoFs, self.ndim).
        mapped : bool, optional
            Whether mapping is applied to node positions before applying ICs.
            The default is True.

        Returns
        -------
        None.

        """
        nDoFs = self.nDoFs
        self.uTime = 0.0
        if isinstance(u0, np.ndarray) and u0.shape == (nDoFs,):
            self.u0 = u0
            self.u = u0.copy()
            self.u0func = None
        elif callable(u0):
            self.u0func = u0
            if mapped:
                self.u = u0(self.DoFsMapped)
            else:
                self.u = u0(self.DoFs)
            self.u0 = self.u.copy()
        else:
            raise TypeError(f"u0 must be an array of shape ({nDoFs},) "
                f"or a callable object returning such an array and taking as "
                f"input the array of node coordinates with shape "
                f"({nDoFs}, {self.ndim}).")

        # pre-allocate arrays for constructing matrix equation for uI
        # this is the maximum possibly required size; not all will be used
        nMaxEntries = int(self.boundary.volume * self.NX * self.NY * nDoFs)
        data = np.empty(nMaxEntries)
        indices = np.empty(nMaxEntries, dtype='uint32')
        indptr = np.empty(self.nNodes+1, dtype='uint32')
        index = 0
        for iN, node in enumerate(self.DoFs):
            inds, phis = self.phi(node)
            nEntries = len(inds)
            data[index:index+nEntries] = phis
            indices[index:index+nEntries] = inds
            indptr[iN] = index
            index += nEntries
        indptr[-1] = index
        print(f"{index}/{nMaxEntries} used for u0")
        A = sp.csr_matrix( (data[0:index], indices[0:index], indptr),
                           shape=(self.nNodes, self.nNodes) )
        self.uI = sp_la.spsolve(A, self.u)

    def computeSpatialDiscretization(self, f=None, NQX=1, NQY=None, Qord=2,
            quadType='gauss', massLumping=False, **kwargs):
        """Assemble the system discretization matrices K, A, M in CSR format.

        K is the stiffness matrix from the diffusion term
        A is the advection matrix
        M is the mass matrix from the time derivative

        Parameters
        ----------
        f : {callable, None}, optional
            Forcing function. Must take 2D array of points and return 1D array.
            The default is None.
        NQX : int, optional
            Number of quadrature cell divisions between FCI planes.
            The default is 1.
        NQY : {int, None}, optional
            Number of quadrature cell divisions in y-direction.
            The default is None, which sets NQY = NY.
        Qord : int, optional
            Number of quadrature points in each grid cell along one dimension.
            The default is 2.
        quadType : string, optional
            Type of quadrature to be used. Must be either 'gauss' or 'uniform'.
            Produces either Gauss-Legendre or Newton-Cotes type points/weights.
            The default is 'gauss'.
        massLumping : bool, optional
            Determines whether mass-lumping is used to calculate M matrix.
            The default is False.

        Returns
        -------
        None.

        """
        self.vci = None
        self.vci_solver = None
        ndim = self.ndim
        nDoFs = self.nDoFs
        NX = self.NX
        NY = self.NY
        ymax = self.ymax
        if NQY is None:
            NQY = NY
        self.f = f
        self.NQX = NQX
        self.NQY = NQY
        self.Qord = Qord
        self.quadType = quadType
        self.massLumping = massLumping
        # pre-allocate arrays for stiffness matrix triplets
        nQuads = NQX * NQY * Qord**2
        nMaxEntries = int((nDoFs * self.boundary.volume)**2 * nQuads * NX)
        Kdata = np.zeros(nMaxEntries)
        Adata = np.zeros(nMaxEntries)
        if not massLumping:
            Mdata = np.zeros(nMaxEntries)
        row_ind = np.zeros(nMaxEntries, dtype='uint32')
        col_ind = np.zeros(nMaxEntries, dtype='uint32')
        self.b = np.zeros(nDoFs)
        self.u_weights = np.zeros(nDoFs)

        ##### compute spatial discretizaton
        index = 0
        for iPlane in range(NX):
            dx = self.dx[iPlane]
            ##### generate quadrature points
            if quadType.lower() in ('gauss', 'g', 'gaussian'):
                offsets, weights = roots_legendre(Qord)
            elif quadType.lower() in ('uniform', 'u'):
                offsets = np.linspace(1/Qord - 1, 1 - 1/Qord, Qord)
                weights = np.repeat(2/Qord, Qord)
            offsets = (offsets * dx * 0.5 / NQX, offsets * ymax * 0.5 / NQY)
            weights = (weights * dx * 0.5 / NQX, weights * ymax * 0.5 / NQY)
            quads = ( np.indices([NQX, NQY], dtype='float').T.
                      reshape(-1, ndim) + 0.5 ) * [dx/NQX, ymax/NQY]
            quadWeights = np.repeat(1., len(quads))
            for i in range(ndim):
                quads = np.concatenate(
                    [quads + offset*np.eye(ndim)[i] for offset in offsets[i]] )
                quadWeights = np.concatenate(
                    [quadWeights * weight for weight in weights[i]] )

            quads += [self.nodeX[iPlane], 0]

            for iQ, quad in enumerate(quads):
                inds, phis, gradphis = self.dphi(quad)
                quadWeight = quadWeights[iQ]
                if f is not None:
                    self.b[inds] += quadWeight * f(quad) * phis
                self.u_weights[inds] += quadWeight * phis
                nEntries = len(inds)**2
                Kdata[index:index+nEntries] = quadWeight * \
                    np.ravel( gradphis @ (self.diffusivity @ gradphis.T) )
                Adata[index:index+nEntries] = quadWeight * \
                    np.ravel( np.outer(np.dot(gradphis, self.velocity), phis) )
                if not massLumping:
                    Mdata[index:index+nEntries] = quadWeight * \
                        np.ravel( np.outer(phis, phis) )
                row_ind[index:index+nEntries] = np.repeat(inds, len(inds))
                col_ind[index:index+nEntries] = np.tile(inds, len(inds))
                index += nEntries

        self.K = sp.csr_matrix( (Kdata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
        self.A = sp.csr_matrix( (Adata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
        if massLumping:
            self.M = sp.diags(self.u_weights, format='csr')
        else:
            self.M = sp.csr_matrix( (Mdata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )

    def computeSpatialDiscretizationLinearVCI(self, f=None, NQX=1, NQY=None,
            Qord=2, quadType='gauss', massLumping=False, **kwargs):
        """Assemble the system discretization matrices K, A, M in CSR format.
        Implements linear variationally consistent integration using assumed
        strain method of Chen2013 https://doi.org/10.1002/nme.4512

        K is the stiffness matrix from the diffusion term
        A is the advection matrix
        M is the mass matrix from the time derivative

        Parameters
        ----------
        f : {callable, None}, optional
            Forcing function. Must take 2D array of points and return 1D array.
            The default is None.
        NQX : int, optional
            Number of quadrature cell divisions between FCI planes.
            The default is 1.
        NQY : {int, None}, optional
            Number of quadrature cell divisions in y-direction.
            The default is None, which sets NQY = NY.
        Qord : int, optional
            Number of quadrature points in each grid cell along one dimension.
            The default is 2.
        quadType : string, optional
            Type of quadrature to be used. Must be either 'gauss' or 'uniform'.
            Produces either Gauss-Legendre or Newton-Cotes type points/weights.
            The default is 'gauss'.
        massLumping : bool, optional
            Determines whether mass-lumping is used to calculate M matrix.
            The default is False.

        Returns
        -------
        None.

        """
        self.vci = 'VC1 (assumed strain)'
        self.vci_solver = None
        ndim = self.ndim
        nDoFs = self.nDoFs
        NX = self.NX
        NY = self.NY
        if NQY is None:
            NQY = NY
        self.f = f
        self.NQX = NQX
        self.NQY = NQY
        self.Qord = Qord
        self.quadType = quadType
        self.massLumping = massLumping
        # pre-allocate arrays for stiffness matrix triplets
        nQuads = NQX * NQY * Qord**2
        nMaxEntries = int((nDoFs * self.boundary.volume)**2 * nQuads * NX)
        Kdata = np.zeros(nMaxEntries)
        Adata = np.zeros(nMaxEntries)
        if not massLumping:
            Mdata = np.zeros(nMaxEntries)
        row_ind = np.zeros(nMaxEntries, dtype='int')
        col_ind = np.zeros(nMaxEntries, dtype='int')
        self.b = np.zeros(nDoFs)
        self.u_weights = np.zeros(nDoFs)

        self.store = []
        self.areas = np.zeros(nDoFs)
        self.xis = np.zeros((self.nDoFs, self.ndim))

        ##### compute spatial discretizaton
        for iPlane in range(NX):
            dx = self.dx[iPlane]
            ##### generate quadrature points
            if quadType.lower() in ('gauss', 'g', 'gaussian'):
                offsets, weights = roots_legendre(Qord)
            elif quadType.lower() in ('uniform', 'u'):
                offsets = np.linspace(1/Qord - 1, 1 - 1/Qord, Qord)
                weights = np.repeat(2/Qord, Qord)
            offsets = (offsets * dx * 0.5 / NQX, offsets * 0.5 / NQY)
            weights = (weights * dx * 0.5 / NQX, weights * 0.5 / NQY)
            quads = ( np.indices([NQX, NQY], dtype='float').T.
                      reshape(-1, ndim) + 0.5 ) * [dx/NQX, 1/NQY]
            quadWeights = np.repeat(1., len(quads))
            for i in range(ndim):
                quads = np.concatenate(
                    [quads + offset*np.eye(ndim)[i] for offset in offsets[i]] )
                quadWeights = np.concatenate(
                    [quadWeights * weight for weight in weights[i]] )
            quads += [self.nodeX[iPlane], 0]

            for iQ, quad in enumerate(quads):
                inds, phis, gradphis = self.dphi(quad)
                quadWeight = quadWeights[iQ]
                self.store.append((inds, phis, gradphis, quadWeight))
                self.areas[inds] += quadWeight
                self.xis[inds] -= gradphis * quadWeight
                if f is not None:
                    self.b[inds] += quadWeight * f(quad) * phis

        self.gradphiSumsOld = -self.xis.copy()
        self.gradphiSumsNew = np.zeros((nDoFs, 2))
        self.xis /= self.areas.reshape(-1,1)

        index = 0
        for (inds, phis, gradphis, quadWeight) in self.store:
            testgrads = gradphis + self.xis[inds]
            self.gradphiSumsNew[inds] += testgrads * quadWeight
            self.u_weights[inds] += quadWeight * phis
            nEntries = len(inds)**2
            Kdata[index:index+nEntries] = quadWeight * \
                np.ravel( testgrads @ (self.diffusivity @ gradphis.T) )
            Adata[index:index+nEntries] = quadWeight * \
                np.ravel( np.outer(np.dot(testgrads, self.velocity), phis) )
            if not massLumping:
                Mdata[index:index+nEntries] = quadWeight * \
                    np.ravel( np.outer(phis, phis) )
            row_ind[index:index+nEntries] = np.repeat(inds, len(inds))
            col_ind[index:index+nEntries] = np.tile(inds, len(inds))
            index += nEntries

        self.K = sp.csr_matrix( (Kdata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
        self.A = sp.csr_matrix( (Adata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
        if massLumping:
            self.M = sp.diags(self.u_weights, format='csr')
        else:
            self.M = sp.csr_matrix( (Mdata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )

    def computeSpatialDiscretizationConservativeLinearVCI(self, f=None, NQX=1,
            NQY=None, Qord=2, quadType='gauss', massLumping=False, **kwargs):
        """Assemble the system discretization matrices K, A, M in CSR format.
        Implements linear variationally consistent integration by re-weighting
        the quadrature points.

        K is the stiffness matrix from the diffusion term
        A is the advection matrix
        M is the mass matrix from the time derivative

        Parameters
        ----------
        f : {callable, None}, optional
            Forcing function. Must take 2D array of points and return 1D array.
            The default is None.
        NQX : int, optional
            Number of quadrature cell divisions between FCI planes.
            The default is 1.
        NQY : {int, None}, optional
            Number of quadrature cell divisions in y-direction.
            The default is None, which sets NQY = NY.
        Qord : int, optional
            Number of quadrature points in each grid cell along one dimension.
            The default is 2.
        quadType : string, optional
            Type of quadrature to be used. Must be either 'gauss' or 'uniform'.
            Produces either Gauss-Legendre or Newton-Cotes type points/weights.
            The default is 'gauss'.
        massLumping : bool, optional
            Determines whether mass-lumping is used to calculate M matrix.
            The default is False.

        Returns
        -------
        None.

        """
        self.vci = 'VC1-C (whole domain)'
        ndim = self.ndim
        nDoFs = self.nDoFs
        nNodes = self.nNodes
        NX = self.NX
        NY = self.NY
        if NQY is None:
            NQY = NY
        self.f = f
        self.NQX = NQX
        self.NQY = NQY
        self.Qord = Qord
        self.quadType = quadType
        self.massLumping = massLumping
        # pre-allocate arrays for stiffness matrix triplets
        nQuads = NQX * NQY * Qord**2
        nMaxEntries = int((nDoFs * self.boundary.volume)**2 * nQuads * NX)
        Kdata = np.zeros(nMaxEntries)
        Adata = np.zeros(nMaxEntries)
        if not massLumping:
            Mdata = np.zeros(nMaxEntries)
        row_ind = np.zeros(nMaxEntries, dtype='int')
        col_ind = np.zeros(nMaxEntries, dtype='int')
        self.b = np.zeros(nDoFs)
        self.u_weights = np.zeros(nNodes)

        self.store = []

        nMaxEntries = int(((nDoFs * self.boundary.volume)*2 + 1) * nQuads * NX)
        gd = np.empty(nMaxEntries)
        ri = np.empty(nMaxEntries, dtype='int')
        ci = np.empty(nMaxEntries, dtype='int')

        self.rOld = np.zeros((nNodes, self.ndim, 3))

        ##### compute spatial discretizaton
        index = 0
        for iPlane in range(NX):
            dx = self.dx[iPlane]
            ##### generate quadrature points
            if quadType.lower() in ('gauss', 'g', 'gaussian'):
                offsets, weights = roots_legendre(Qord)
            elif quadType.lower() in ('uniform', 'u'):
                offsets = np.linspace(1/Qord - 1, 1 - 1/Qord, Qord)
                weights = np.repeat(2/Qord, Qord)
            offsets = (offsets * dx * 0.5 / NQX, offsets * 0.5 / NQY)
            weights = (weights * dx * 0.5 / NQX, weights * 0.5 / NQY)
            quads = ( np.indices([NQX, NQY], dtype='float').T.
                      reshape(-1, ndim) + 0.5 ) * [dx/NQX, 1/NQY]
            quadWeights = np.repeat(1., len(quads))
            for i in range(ndim):
                quads = np.concatenate(
                    [quads + offset*np.eye(ndim)[i] for offset in offsets[i]] )
                quadWeights = np.concatenate(
                    [quadWeights * weight for weight in weights[i]] )

            quads += [self.nodeX[iPlane], 0]

            for iQ, quad in enumerate(quads):
                inds, phis, gradphis = self.dphi(quad)
                quadWeight = quadWeights[iQ]
                disps = quad - self.nodes[inds]
                rNews = np.apply_along_axis(lambda x: np.outer(x[0:2],
                    x[2:4]), 1, np.hstack((gradphis, disps)))
                self.store.append((inds, phis, gradphis, quadWeight, rNews))
                self.rOld[inds,:,0] -= gradphis * quadWeight
                self.rOld[inds,0,1] -= phis * quadWeight
                self.rOld[inds,1,2] -= phis * quadWeight
                self.rOld[inds,:,1:3] -= rNews * quadWeight
                if f is not None:
                    self.b[inds] += f(quad) * phis * quadWeight
                nEntries = 2*len(inds)
                gd[index:index+nEntries] = gradphis.ravel()
                ri[index:index+nEntries:2] = inds
                ri[index+1:index+nEntries:2] = inds + nDoFs
                ci[index:index+nEntries] = iQ + iPlane*nQuads
                index += nEntries

        gd[index:index + nQuads*NX] = 1.0
        ri[index:index + nQuads*NX] = 2*nDoFs
        ci[index:index + nQuads*NX] = np.arange(nQuads * NX)
        index += nQuads * NX

        self.gradphiSums = self.rOld[:nDoFs,:,0]
        nConstraints = 2*nDoFs + 1

        ##### Using SuiteSparse min2norm (QR based solver) #####
        G = sp.csc_matrix((gd[:index], (ri[:index], ci[:index])),
                          shape=(np.iinfo('int32').max + 1, nQuads * NX))
        G._shape = (nConstraints, nQuads * NX)
        del gd, ci, ri, offsets, weights, quads, quadWeights
        from timeit import default_timer
        start_time = default_timer()
        rhs = np.append(self.gradphiSums.T.ravel(), 0.)
        self.xi = (ssqr.min2norm(G, rhs).ravel(), 0)
        print(f'xi solve time = {default_timer()-start_time} s')
        self.vci_solver = 'ssqr.min2norm'

        # ##### Using scipy.sparse.linalg, much slower, but uses less memory #####
        # self.G = sp.csr_matrix((gd[:index], (ri[:index], ci[:index])),
        #                         shape=(nConstraints, nQuads * NX))
        # rhs = np.append(self.gradphiSums.T.ravel(), 0.)
        # v0 = np.zeros(nQuads * NX)
        # maxit = nQuads * NX
        # # tol = np.finfo(float).eps
        # tol = 1e-10
        # from timeit import default_timer
        # start_time = default_timer()
        # # self.xi = sp_la.lsmr(self.G, rhs, x0=v0, atol=tol, btol=tol, maxiter=maxit)
        # self.xi = sp_la.lsqr(self.G, rhs, x0=v0, atol=tol, btol=tol, iter_lim=maxit)
        # print(f'xi solve time = {default_timer()-start_time} s')
        # self.vci_solver = 'scipy.sparse.linalg.lsqr'

        self.rNew = np.zeros((nNodes, self.ndim, 3))

        index = 0
        for iQ, (inds, phis, gradphis, quadWeight, rNews) in enumerate(self.store):
            quadWeight += self.xi[0][iQ]
            disps = quad - self.nodes[inds]
            self.rNew[inds,:,0] -= gradphis * quadWeight
            self.rNew[inds,0,1] -= phis * quadWeight
            self.rNew[inds,1,2] -= phis * quadWeight
            self.rNew[inds,:,1:3] -= rNews * quadWeight
            self.u_weights[inds] += quadWeight * phis
            nEntries = len(inds)**2
            Kdata[index:index+nEntries] = quadWeight * \
                np.ravel( gradphis @ (self.diffusivity @ gradphis.T) )
            Adata[index:index+nEntries] = quadWeight * \
                np.ravel( np.outer(np.dot(gradphis, self.velocity), phis) )
            if not massLumping:
                Mdata[index:index+nEntries] = quadWeight * \
                    np.ravel( np.outer(phis, phis) )
            row_ind[index:index+nEntries] = np.repeat(inds, len(inds))
            col_ind[index:index+nEntries] = np.tile(inds, len(inds))
            index += nEntries

        self.gradphiSumsNew = self.rNew[:nDoFs,:,0]

        self.K = sp.csr_matrix( (Kdata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
        self.A = sp.csr_matrix( (Adata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
        if massLumping:
            self.M = sp.diags(self.u_weights, format='csr')
        else:
            self.M = sp.csr_matrix( (Mdata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )

    def initializeTimeIntegrator(self, integrator, dt, P='ilu', **kwargs):
        """Initialize and register the time integration scheme to be used.

        Parameters
        ----------
        integrator : {Integrator (object or subclass type), string}
            Integrator object or string specifiying which scheme is to be used.
            If a string, must be one of 'LowStorageRK' ('RK' or 'LSRK'),
            'BackwardEuler' ('BE'), or 'CrankNicolson' ('CN').
        dt : float
            Time interval between each successive timestep.
        P : {string, scipy.sparse.linalg.LinearOperator, None}, optional
            Which preconditioning method to use. P can be a LinearOperator to
            directly specifiy the preconditioner to be used. Otherwise it must
            be one of 'jacobi', 'ilu', or None. The default is 'ilu'.
        **kwargs
            Used to specify optional arguments for the time integrator.
            Will be passed to scipy.sparse.linalg.spilu if 'ilu' is used, or
            can be used to specify betas for LowStorageRK schemes.

        Returns
        -------
        None.

        """
        if isinstance(integrator, integrators.Integrator):
            self.integrator = integrator
            return
        if isinstance(integrator, str):
            if integrator.lower() in ('backwardeuler', 'be'):
                Type = integrators.BackwardEuler
            elif integrator.lower() in ('cranknicolson', 'cn'):
                Type = integrators.CrankNicolson
            elif integrator.lower() in ('lowstoragerk', 'rk', 'lsrk'):
                Type = integrators.LowStorageRK
        else: # if integrator not an Integrator object or string, assume it's a type
            Type = integrator
        # Instantiate and store the integrator object
        try:
            self.integrator = Type(self, self.A - self.K, self.M, dt, P, **kwargs)
        except:
            raise TypeError("Unable to instantiate integrator of type "
                f"{repr(Type)}. Should be a string containing one of "
                "'LowStorageRK' ('RK' or 'LSRK'), CrankNicolson ('CN'), or "
                "'BackwardEuler' ('BE'), a type derived from "
                "integrators.Integrator, or an object of such a type.")

    def step(self, nSteps=1, **kwargs):
        """Advance the simulation a given number of timesteps.

        Parameters
        ----------
        nSteps : int, optional
            Number of timesteps to compute. The default is 1.
        **kwargs
            Used to specify optional arguments passed to the linear solver.
            Note that kwargs["M"] will be overwritten, instead use
            self.precondition(...) to generate or specify a preconditioner.

        Returns
        -------
        None.

        """
        self.integrator.step(nSteps, **kwargs)

    def solve(self):
        """Reconstruct the final solution vector, u, from the shape functions.

        Returns
        -------
        None.

        """
        # self.uTime = self.integrator.time
        self.u = np.empty(self.nNodes)
        for iN, node in enumerate(self.nodes):
            indices, phis = self.phi(node)
            self.u[iN] = self.uI[indices] @ phis

    def generatePlottingPoints(self, nx=1, ny=1):
        """Generate set of interpolation points to use for plotting.

        Parameters
        ----------
        nx : int, optional
            Number of points per grid division in the x-direction.
            The default is 1.
        ny : int, optional
            Number of points per grid division in the y-direction.
            The default is 1.

        Returns
        -------
        None.

        """
        NX = self.NX
        NY = self.NY
        self.phiPlot = []
        self.indPlot = []
        self.X = []

        if self.boundary.name == 'periodic':
            self.uPlot = self.uI
        else:
            self.uPlot = np.concatenate((self.uI, [1.]))

        for iPlane in range(NX):
            points = np.indices((nx, NY*ny + 1), dtype='float') \
                .reshape(self.ndim, -1).T * [self.dx[iPlane]/nx, 1/(NY*ny)]
            points[:,0] += self.nodeX[iPlane]
            self.X.append(points[:,0])
            for iP, point in enumerate(points):
                inds, phis = self.phi(point)
                self.phiPlot.append(phis)
                inds[inds < 0] = -1
                self.indPlot.append(inds)
        # Deal with right boundary
        points = np.hstack((np.full((NY*ny + 1, 1), self.xmax), points[0:NY*ny + 1,1:2]))
        for point in points:
                inds, phis = self.phi(point)
                self.phiPlot.append(phis)
                inds[inds < 0] = -1
                self.indPlot.append(inds)

        self.X.append(np.full(NY*ny+1, self.xmax))
        self.X = np.concatenate(self.X)
        self.Y = np.tile(points[0:NY*ny + 1,1], NX*nx + 1)
        self.U = np.empty(len(self.X))
        for i, x in enumerate(self.X):
            self.U[i] = np.sum(self.phiPlot[i] * self.uPlot[self.indPlot[i]])

    def computePlottingSolution(self):
        """Compute interpolated solution at the plotting points.

        Returns
        -------
        None.

        """
        self.uPlot[0:self.nDoFs] = self.uI
        for i, x in enumerate(self.X):
            self.U[i] = np.sum(self.phiPlot[i] * self.uPlot[self.indPlot[i]])
