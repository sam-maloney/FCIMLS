# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:47:07 2020

@author: Samuel A. Maloney

"""

from timeit import default_timer
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
    nNodes : int
        Number of unique nodal points in the simulation domain.
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
    b : numpy.ndarray, shape=(nNodes,)
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
            quadType='gauss', massLumping=False, vci='linear', **kwargs)
        Assemble the system discretization matrices K, A, M in CSR format.
    initializeTimeIntegrator(self, integrator, dt, P='ilu', **kwargs):
        Initialize and register the time integration scheme to be used.
    step(self, nSteps=1, **kwargs):
        Advance the simulation a given number of timesteps.
    solve(self)
        Reconstruct the final solution vector, u, from the shape functions.
    generatePlottingPoints(self, nx=1, ny=1):
        Generate set of interpolation points to use for plotting.
    computePlottingSolution(self):
        Compute interpolated solution at the plotting points.

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
            dx = xmax/NX
            self.nodeX = dx*np.arange(0, NX+1, 1)
            px *= dx
            self.nodeX[1:-1] += rng.uniform(-px, px, self.nodeX[1:-1].shape)
        dy = ymax/NY
        self.nodeY = np.tile(dy*np.arange(0, NY+1, 1), NX+1).reshape(NX+1,-1)
        py *= dy
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
            self.boundary = boundaries.DirichletBoundary(self, *boundary[1])
        else:
            raise TypeError(f"Unkown boundary condition: {boundary}")
        self.nodes = self.boundary.computeNodes()
        self.nNodes = self.boundary.nNodes
        self.nodesMapped = self.nodes.copy()
        self.nodesMapped[:,1] = self.boundary.mapping(self.nodes, 0)
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
        ##### Centred-and-scaled MLS #####
        # --------------------------------------
        #     compute the moment matrix A(x)
        # --------------------------------------
        indices, w, displacements = self.boundary.w(point)
        p = self.basis(displacements)
        A = w*p.T@p
        # --------------------------------------
        #      compute vector c(x) and phi
        # --------------------------------------
        # A(x)c(x) = p(x)
        # Backward substitution for c(x) using LU factorization for A
        p0 = self.basis.p0()
        lu, piv = la.lu_factor(A, overwrite_a=True, check_finite=False)
        c = la.lu_solve((lu, piv), p0, overwrite_b=True, check_finite=False)
        phis = c @ p.T * w
        if indices.size < self.basis.size:
            print(f"Error: insufficient coverage at p = {point}")
        if np.abs(phis.sum() - 1) > 1e-10:
            print(f"Error: phis not partition of unity at p = {point}")
        return indices, phis

        # ##### Standard MLS (Nguyen2008) #####
        # # --------------------------------------
        # #     compute the moment matrix A(x)
        # # --------------------------------------
        # indices, w, _ = self.boundary.w(point)
        # displacements = self.boundary.computeDisplacements(point, indices)
        # p = self.basis(point - displacements)
        # A = w*p.T@p
        # # --------------------------------------
        # #      compute vector c(x) and phi
        # # --------------------------------------
        # # A(x)c(x) = p(x)
        # # Backward substitution for c(x) using LU factorization for A
        # p_x = self.basis(point)[0]
        # lu, piv = la.lu_factor(A, overwrite_a=True, check_finite=False)
        # c = la.lu_solve((lu, piv), p_x, overwrite_b=True, check_finite=False)
        # phi = c @ p.T * w
        # return indices, phi

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
        ##### Centred-and-scaled MLS #####
        # --------------------------------------
        #     compute the moment matrix A(x)
        # --------------------------------------
        indices, w, gradw, displacements = self.boundary.dw(point)
        p = self.basis(displacements)
        dp = self.basis.dp(displacements) * \
             self.boundary.rsupport.reshape(self.ndim,-1)
        # re-align gradients to global x-coordinate
        gradw[:,0] -= self.mapping.deriv(point)*gradw[:,1]
        dp[:,0] -= self.mapping.deriv(point)*dp[:,1]
        wp = w*p.T
        A = wp@p
        dA = [gradw[:,i]*p.T@p + dp[:,i].T@wp.T + wp@dp[:,i]
              for i in range(self.ndim)]
        # --------------------------------------
        #      compute matrix c
        # --------------------------------------
        # A(x)c(x) = p(x)
        # A(x)c_k(x) = b_k(x)
        # Backward substitutions, once for c(x), ndim times for c_k(x)
        # Using LU factorization for A
        p0 = self.basis.p0()
        lu, piv = la.lu_factor(A, overwrite_a=True, check_finite=False)
        c = np.empty((self.ndim + 1, self.basis.size))
        c[0] = la.lu_solve((lu, piv), p0, overwrite_b=True, check_finite=False)
        for i in range(self.ndim):
            c[i+1] = la.lu_solve((lu, piv), -dA[i]@c[0], check_finite=False)
        # --------------------------------------
        #       compute phi and gradphi
        # --------------------------------------
        cp = c[0] @ p.T
        phis = cp * w
        gradphis = ( c[1:]@p.T*w + cp*gradw.T).T
        gradphis[:,0] += (c[0]@dp[:,0].T*w).T
        gradphis[:,1] += (c[0]@dp[:,1].T*w).T
        if indices.size < self.basis.size:
            print(f"Error: insufficient coverage at p = {point}")
        if np.abs(phis.sum() - 1) > 1e-10:
            print(f"Error: phis not partition of unity at p = {point}")
        # np.testing.assert_allclose(gradphis.sum(axis=0), (0,0), atol=1e-10)
        return indices, phis, gradphis

        # ##### Standard MLS (Nguyen2008) #####
        # # --------------------------------------
        # #     compute the moment matrix A(x)
        # # --------------------------------------
        # indices, w, gradw, _ = self.boundary.dw(point)
        # displacements = self.boundary.computeDisplacements(point, indices)
        # p = self.basis(point - displacements)
        # A = w*p.T@p
        # # re-align gradient to global x-coordinate
        # gradw[:,0] -= self.mapping.deriv(point)*gradw[:,1]
        # dA = [gradw[:,i]*p.T@p for i in range(self.ndim)]
        # # --------------------------------------
        # #         compute matrix c
        # # --------------------------------------
        # # A(x)c(x)   = p(x)
        # # A(x)c_k(x) = b_k(x)
        # # Backward substitutions, once for c(x), ndim times for c_k(x)
        # # Using LU factorization for A
        # p_x = self.basis(point)[0]
        # lu, piv = la.lu_factor(A, check_finite=False)
        # c = np.empty((self.ndim + 1, self.basis.size))
        # c[0] = la.lu_solve((lu, piv), p_x, check_finite=False)
        # dp = self.basis.dp(point)
        # for i in range(self.ndim):
        #     c[i+1] = la.lu_solve( (lu, piv), (dp[i] - dA[i]@c[0]),
        #                           check_finite=False )
        # # --------------------------------------
        # #       compute phi and gradphi
        # # --------------------------------------
        # cp = c[0] @ p.T
        # phis = cp * w
        # gradphis = ( c[1 : self.ndim + 1]@p.T*w + cp*gradw.T).T
        # return indices, phis, gradphis

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
        # TODO: re-align grad2w to global x-coordinate
        # --------------------------------------
        #     compute the moment matrix A(x)
        # --------------------------------------
        indices, w, gradw, grad2w = self.boundary.d2w(point)
        displacements = self.boundary.computeDisplacements(point, indices)
        p = self.basis(point - displacements)
        A = w*p.T@p
        # re-align gradient to global x-coordinate
        gradw[:,0] -= self.mapping.deriv(point)*gradw[:,1]
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
        return indices, phis, grad2phis

    def setInitialConditions(self, u0, mapped=True):
        """Initialize the nodal coefficients for the given IC.

        Parameters
        ----------
        u0 : {numpy.ndarray, callable}
            Initial conditions for the simulation.
            Must be an array of shape (self.nNodes,) or a callable object
            returning such an array and taking as input the array of node
            coordinates with shape (self.nNodes, self.ndim).
        mapped : bool, optional
            Whether mapping is applied to node positions before applying ICs.
            The default is True.

        Returns
        -------
        None.

        """
        nNodes = self.nNodes
        self.uTime = 0.0
        if isinstance(u0, np.ndarray) and u0.shape == (nNodes,):
            self.u0 = u0
            self.u = u0.copy()
            self.u0func = None
        elif callable(u0):
            self.u0func = u0
            if mapped:
                self.u = u0(self.nodesMapped)
            else:
                self.u = u0(self.nodes)
            self.u0 = self.u.copy()
        else:
            raise TypeError(f"u0 must be an array of shape ({nNodes},) "
                f"or a callable object returning such an array and taking as "
                f"input the array of node coordinates with shape "
                f"({nNodes}, {self.ndim}).")

        # pre-allocate arrays for constructing matrix equation for uI
        # this is the maximum possibly required size; not all will be used
        nMaxEntries = int(self.boundary.volume * self.NX * self.NY * nNodes)
        data = np.empty(nMaxEntries)
        indices = np.empty(nMaxEntries, dtype='uint32')
        indptr = np.empty(self.nNodes+1, dtype='uint32')
        index = 0
        for iN, node in enumerate(self.nodes):
            inds, phis = self.phi(node)
            nEntries = inds.size
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
            quadType='gauss', massLumping=False, vci='linear', **kwargs):
        """Assemble the system discretization matrices K, A, M in CSR format.
        Implements linear/quadratic variationally consistent integration using
        assumed strain method of Chen2013 https://doi.org/10.1002/nme.4512

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
        vci : {int, string, None}, optional
            Order of VCI correction to apply. If int must be 1 or 2, if string
            must be in ['linear', 'quadratic']. The Default is 'linear'.

        Returns
        -------
        None.

        """
        if vci in [0, None]:
            self.vci = None
            vci = None
        elif vci in [1, 'linear', 'Linear', 'l', 'L']:
            self.vci = 'VC1 (assumed strain)'
            vci = 1
        elif vci in [2, 'quadratic', 'Quadratic', 'q', 'Q']:
            self.vci = 'VC2 (assumed strain)'
            vci = 2
        else:
             raise ValueError('Unknown VCI order vci={vci}')
        self.vci_solver = None
        ndim = self.ndim
        nNodes = self.nNodes
        NX = self.NX
        NY = self.NY
        if NQY is None:
            NQY = NY
        self.f = f
        self.NQX = NQX
        self.NQY = NQY
        self.Qord = Qord
        self.massLumping = massLumping
        # pre-allocate arrays for operator matrix triplets
        nQuads = NX * NQX * NQY * Qord**2
        self.nQuads = nQuads
        nMaxEntries = int((nNodes * self.boundary.volume)**2 * nQuads)
        Kdata = np.empty(nMaxEntries)
        Adata = np.empty(nMaxEntries)
        if not massLumping:
            Mdata = np.empty(nMaxEntries)
        row_ind = np.empty(nMaxEntries, dtype='int')
        col_ind = np.empty(nMaxEntries, dtype='int')
        self.b = np.zeros(nNodes)
        self.u_weights = np.zeros(nNodes)

        storage = []

        if vci == 2:
            A = np.zeros((self.nNodes, 3, 3))
            self.rOld, opMatBints = self.boundary.computeBoundaryIntegrals(vci)
            self.rNew = self.rOld.copy()
            self.gradphiSumsOld = self.rOld[:,:,0]
            self.gradphiSumsNew = self.rNew[:,:,0]
            xis = np.empty((self.nNodes, self.ndim, 3))
        else: # if vci == 1 or None
            self.gradphiSumsOld, opMatBints = self.boundary.computeBoundaryIntegrals(vci=1)
            # self.gradphiSumsNew = self.gradphiSumsOld
            # if vci == 1:
            #     areas = np.zeros(nNodes)
            #     self.gradphiSumsNew = self.gradphiSumsOld.copy()
            areas = np.zeros(nNodes)
            self.gradphiSumsNew = self.gradphiSumsOld.copy()

        ##### compute spatial discretizaton
        for iPlane in range(NX):
            dx = self.dx[iPlane]
            ##### generate quadrature points
            if quadType.lower()[0] == 'g':
                self.quadType = 'Gauss-Legendre'
                offsets, weights = roots_legendre(Qord)
            elif quadType.lower()[0] == 'u':
                self.quadType = 'uniform'
                offsets = np.arange(1/Qord - 1, 1, 2/Qord)
                weights = np.full(Qord, 2/Qord)
            offsets = (offsets * dx * 0.5 / NQX, offsets * 0.5 / NQY)
            weights = (weights * dx * 0.5 / NQX, weights * 0.5 / NQY)
            quads = ( np.indices([NQX, NQY], dtype='float').T.
                      reshape(-1, ndim) + 0.5 ) * [dx/NQX, 1/NQY]
            quadWeights = np.ones(len(quads))
            for i in range(ndim):
                quads = np.concatenate(
                    [quads + offset*np.eye(ndim)[i] for offset in offsets[i]] )
                quadWeights = np.concatenate(
                    [quadWeights * weight for weight in weights[i]] )
            quads += [self.nodeX[iPlane], 0]

            for iQ, quad in enumerate(quads):
                inds, phis, gradphis = self.dphi(quad)
                quadWeight = quadWeights[iQ]
                self.gradphiSumsOld[inds] -= gradphis * quadWeight
                if vci == 2:
                    disps = self.boundary.computeDisplacements(quad, inds)
                    storage.append((inds, phis, gradphis, quadWeight, disps))
                    P = np.hstack((np.ones((inds.size, 1)), disps))
                    A[inds] += quadWeight * \
                        np.apply_along_axis(lambda x: np.outer(x,x), 1, P)
                    self.rOld[inds,0,1] -= phis * quadWeight
                    self.rOld[inds,1,2] -= phis * quadWeight
                    self.rOld[inds,:,1:] -= np.apply_along_axis(lambda x: np.outer(x[0:2],
                        x[2:4]), 1, np.hstack((gradphis, disps))) * quadWeight
                else: # if vci == 1 or None
                    storage.append((inds, phis, gradphis, quadWeight))
                    # if vci == 1:
                    #     areas[inds] += quadWeight
                    areas[inds] += quadWeight
                if f is not None:
                    self.b[inds] += quadWeight * f(quad) * phis
        
        del offsets, weights, quads

        if vci == 1:
            xis = self.gradphiSumsOld / areas.reshape(-1,1)
            del areas
        elif vci == 2:
            for i, row in enumerate(A):
                lu, piv = la.lu_factor(A[i], True, False)
                for j in range(self.ndim):
                    xis[i,j] = la.lu_solve((lu, piv), self.rOld[i,j], 0, False, False)            

        index = 0
        for items in storage:
            if vci == 2:
                inds, phis, gradphis, quadWeight, disps = items
                testgrads = ( gradphis + xis[inds,:,0] +
                    xis[inds,:,1]*disps[:,0:1] + xis[inds,:,2]*disps[:,1:2] )
                self.rNew[inds,0,1] -= phis * quadWeight
                self.rNew[inds,1,2] -= phis * quadWeight
                self.rNew[inds,:,1:] -= np.apply_along_axis(lambda x: np.outer(x[0:2],
                    x[2:4]), 1, np.hstack((testgrads, disps))) * quadWeight
            else: # if vci == 1 or None
                inds, phis, gradphis, quadWeight = items
                if vci == 1:
                    testgrads = gradphis + xis[inds]
                else: # if vci == None
                    testgrads = gradphis
            self.gradphiSumsNew[inds] -= testgrads * quadWeight
            self.u_weights[inds] += quadWeight * phis
            nEntries = inds.size**2
            Kdata[index:index+nEntries] = quadWeight * \
                np.ravel( testgrads @ (self.diffusivity @ gradphis.T) )
            Adata[index:index+nEntries] = quadWeight * \
                np.ravel( np.outer(np.dot(testgrads, self.velocity), phis) )
            if not massLumping:
                Mdata[index:index+nEntries] = quadWeight * \
                    np.ravel( np.outer(phis, phis) )
            row_ind[index:index+nEntries] = np.repeat(inds, inds.size)
            col_ind[index:index+nEntries] = np.tile(inds, inds.size)
            index += nEntries
            
        del storage
        
        Kdata = np.concatenate((Kdata[:index], opMatBints[0]))
        Adata = np.concatenate((Adata[:index], opMatBints[1]))
        row_ind = np.concatenate((row_ind[:index], opMatBints[2]))
        col_ind = np.concatenate((col_ind[:index], opMatBints[3]))
        self.K = sp.csr_matrix( (Kdata, (row_ind, col_ind)),
                                shape=(nNodes, nNodes) )
        self.A = sp.csr_matrix( (Adata, (row_ind, col_ind)),
                                shape=(nNodes, nNodes) )
        if massLumping:
            self.M = sp.diags(self.u_weights, format='csr')
        else:
            self.M = sp.csr_matrix( (Mdata[:index], (row_ind[:index], col_ind[:index])),
                                shape=(nNodes, nNodes) )

    def computeSpatialDiscretizationConservativeVCI(self, f=None, NQX=1,
            NQY=None, Qord=2, quadType='gauss', massLumping=False,
            vci='linear', **kwargs):
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
        vci : {int, string}, optional
            Order of VCI correction to apply. If int must be 1 or 2, if string
            must be in ['linear', 'quadratic']. The Default is 'linear'.

        Returns
        -------
        None.

        """
        if vci in [1, 'linear', 'Linear', 'l', 'L']:
            self.vci = 'VC1-C (whole domain)'
            vci = 1
        elif vci in [2, 'quadratic', 'Quadratic', 'q', 'Q']:
            self.vci = 'VC2-C (whole domain)'
            vci = 2
        else:
             raise ValueError('Unknown VCI order vci={vci}')
        ndim = self.ndim
        nNodes = self.nNodes
        nNodes = self.nNodes
        NX = self.NX
        NY = self.NY
        if NQY is None:
            NQY = NY
        self.f = f
        self.NQX = NQX
        self.NQY = NQY
        self.Qord = Qord
        self.massLumping = massLumping
        # pre-allocate arrays for operator matrix triplets
        nQuadsPerPlane = NQX * NQY * Qord**2
        nQuads = nQuadsPerPlane * NX
        self.nQuads = nQuads
        nMaxEntries = int((nNodes * self.boundary.volume)**2 * nQuads)
        Kdata = np.empty(nMaxEntries)
        Adata = np.empty(nMaxEntries)
        if not massLumping:
            Mdata = np.empty(nMaxEntries)
        row_ind = np.empty(nMaxEntries, dtype='int')
        col_ind = np.empty(nMaxEntries, dtype='int')
        self.b = np.zeros(nNodes)
        self.u_weights = np.zeros(nNodes)

        storage = []
        quadWeightsList = []

        if vci == 1:
            nConstraintsPerNode = 2
        elif vci == 2:
            nConstraintsPerNode = 6
        nConstraints = nConstraintsPerNode * nNodes
        if nQuads < nConstraints:
            print('Warning: less quadrature points than VCI constraints')
        nMaxEntries = int(nQuads * (nConstraints * self.boundary.volume + 1))
        gd = np.empty(nMaxEntries)
        ri = np.empty(nMaxEntries, dtype='int')
        ci = np.empty(nMaxEntries, dtype='int')

        vciBints, opMatBints = self.boundary.computeBoundaryIntegrals(vci)
        if vci == 1:
            self.gradphiSumsOld = vciBints.copy()
            self.gradphiSumsNew = self.gradphiSumsOld.copy()
        elif vci == 2:
            self.rOld = vciBints.copy()
            self.rNew = self.rOld.copy()
            self.gradphiSumsOld = self.rOld[:,:,0]
            self.gradphiSumsNew = self.rNew[:,:,0]

        ##### compute spatial discretizaton
        index = 0
        for iPlane in range(NX):
            dx = self.dx[iPlane]
            ##### generate quadrature points
            if quadType.lower()[0] == 'g':
                self.quadType = 'Gauss-Legendre'
                offsets, weights = roots_legendre(Qord)
            elif quadType.lower()[0] == 'u':
                self.quadType = 'uniform'
                offsets = np.arange(1/Qord - 1, 1, 2/Qord)
                weights = np.full(Qord, 2/Qord)
            offsets = (offsets * dx * 0.5 / NQX, offsets * 0.5 / NQY)
            weights = (weights * dx * 0.5 / NQX, weights * 0.5 / NQY)
            quads = ( np.indices([NQX, NQY], dtype='float').T.
                      reshape(-1, ndim) + 0.5 ) * [dx/NQX, 1/NQY]
            quadWeights = np.ones(len(quads))
            for i in range(ndim):
                quads = np.concatenate(
                    [quads + offset*np.eye(ndim)[i] for offset in offsets[i]] )
                quadWeights = np.concatenate(
                    [quadWeights * weight for weight in weights[i]] )

            quads += [self.nodeX[iPlane], 0]

            for iQ, quad in enumerate(quads):
                inds, phis, gradphis = self.dphi(quad)

                nInds = inds.size
                nEntries = 2*nInds
                gd[index:index+nEntries] = gradphis.T.ravel()
                ri[index:index+nInds] = inds
                ri[index+nInds:index+nEntries] = inds + nNodes
                ci[index:index+nConstraintsPerNode*nInds] = iQ + iPlane*nQuadsPerPlane
                index += nEntries

                quadWeight = quadWeights[iQ]
                self.gradphiSumsOld[inds] -= gradphis * quadWeight
                if vci == 1:
                    storage.append((quad, inds, phis, gradphis))
                elif vci == 2:
                    disps = self.boundary.computeDisplacements(quad, inds)
                    rDisps = np.apply_along_axis(lambda x: np.outer(x[0:2],
                        x[2:4]), 1, np.hstack((gradphis, disps)))
                    rDisps[:,0,0] += phis
                    rDisps[:,1,1] += phis
                    self.rOld[inds,:,1:] -= rDisps * quadWeight
                    storage.append((quad, inds, phis, gradphis, rDisps))
                    nEntries = 4*nInds
                    gd[index:index+nEntries] = rDisps.T.ravel()
                    ri[index:index+nInds] = inds + 2*nNodes
                    ri[index+nInds:index+2*nInds] = inds + 3*nNodes
                    ri[index+2*nInds:index+3*nInds] = inds + 4*nNodes
                    ri[index+3*nInds:index+nEntries] = inds + 5*nNodes
                    index += nEntries
                
            quadWeightsList.append(quadWeights)

        # Add final constraint that sum of weights equal domain area
        nConstraints += 1
        self.nConstraints = nConstraints
        gd[index:index + nQuads] = 1.0
        ri[index:index + nQuads] = nConstraintsPerNode*nNodes
        ci[index:index + nQuads] = np.arange(nQuads)
        index += nQuads

        ##### Using SuiteSparse min2norm (QR based solver) #####
        self.vci_solver = 'ssqr.min2norm'
        G = sp.csc_matrix((gd[:index], (ri[:index], ci[:index])),
                          shape=(np.iinfo('int32').max + 1, nQuads))
        G._shape = (nConstraints, nQuads)
        del gd, ci, ri, offsets, weights, quads
        start_time = default_timer()
        # ### solve for quadWeights directly (does not work as well!)
        # rhs = np.append(vciBints.T.ravel(), self.xmax*self.ymax)
        # quadWeights = ssqr.min2norm(G, rhs).ravel()
        ### solve for corrections to quadWeights
        quadWeights = np.concatenate(quadWeightsList)
        if vci == 1:
            rhs = np.append(self.gradphiSumsOld.T.ravel(), 0)
        elif vci == 2:
            rhs = np.append(self.rOld.T.ravel(), 0)
        quadWeightCorrections = ssqr.min2norm(G, rhs).ravel()
        quadWeights += quadWeightCorrections
        print(f'xi solve time = {default_timer()-start_time} s')
        
        # ##### Using sparse_dot_mkl (QR based solver) (not-working) #####
        # self.vci_solver = 'sparse_dot_mkl.sparse_qr_solve_mkl'
        # import sparse_dot_mkl
        # # G = sp.csr_matrix((gd[:index], (ri[:index], ci[:index])),
        # #                   shape=(nConstraints, np.iinfo('int32').max + 1))
        # # G._shape = (nConstraints, nQuads)
        # G = sp.csr_matrix((gd[:index], (ri[:index], ci[:index])),
        #                   shape=(nConstraints, nQuads))
        # del gd, ci, ri, offsets, weights, quads, quadWeights
        # start_time = default_timer()
        # rhs = np.append(vciBints.T.ravel(), self.xmax*self.ymax)
        # quadWeights = sparse_dot_mkl.sparse_qr_solve_mkl(G, rhs).ravel()
        # print(f'xi solve time = {default_timer()-start_time} s')

        # ##### Using scipy.sparse.linalg, much slower, but uses less memory #####
        # G = sp.csr_matrix((gd[:index], (ri[:index], ci[:index])),
        #                         shape=(nConstraints, nQuads))
        # maxit = 100*nQuads
        # tol = 1e-10
        # start_time = default_timer()
        # # ### solve for quadWeights directly (does not work nearly as well!)
        # # rhs = np.append(vciBints.T.ravel(), self.xmax*self.ymax)
        # # # quadWeights = sp_la.lsmr(self.G, rhs, atol=tol, btol=tol, maxiter=maxit)[0]
        # # # self.vci_solver = 'scipy.sparse.linalg.lsmr'
        # # quadWeights = sp_la.lsqr(self.G, rhs, atol=tol, btol=tol, iter_lim=maxit)[0]
        # # self.vci_solver = 'scipy.sparse.linalg.lsqr'
        # ### solve for corrections to quadWeights
        # quadWeights = np.concatenate(quadWeightsList)
        # if vci == 1:
        #     rhs = np.append(self.gradphiSumsOld.T.ravel(), 0)
        # elif vci == 2:
        #     rhs = np.append(self.rOld.T.ravel(), 0)
        # # scale rows of G
        # for i in range(G.shape[0]):
        #     norm = np.sqrt((G.data[G.indptr[i]:G.indptr[i+1]]**2).sum())
        #     G.data[G.indptr[i]:G.indptr[i+1]] /= norm
        #     rhs[i] /= norm
        # # self.vci_solver = 'scipy.sparse.linalg.lsmr'
        # # quadWeightCorrections = sp_la.lsmr(G, rhs, atol=tol, btol=tol, maxiter=maxit)
        # self.vci_solver = 'scipy.sparse.linalg.lsqr'
        # quadWeightCorrections = sp_la.lsqr(G, rhs, atol=tol, btol=tol, iter_lim=maxit)
        # if quadWeightCorrections[1] == 7:
        #     print('Max iterations reached in xi solve')
        # quadWeights += quadWeightCorrections[0]
        # print(f'xi solve time = {default_timer()-start_time} s')
        
        del G, rhs, quadWeightsList
        try:
            del quadWeightCorrections
        except:
            pass

        index = 0
        for iQ, items in enumerate(storage):
            quadWeight = quadWeights[iQ]
            if vci == 1:
                quad, inds, phis, gradphis = items
            if vci == 2:
                quad, inds, phis, gradphis, rDisps = items
                self.rNew[inds,:,1:] -= rDisps * quadWeight
            self.gradphiSumsNew[inds] -= gradphis * quadWeight
            self.u_weights[inds] += quadWeight * phis
            if f is not None:
                self.b[inds] += f(quad) * phis * quadWeight
            nEntries = inds.size**2
            Kdata[index:index+nEntries] = quadWeight * \
                np.ravel( gradphis @ (self.diffusivity @ gradphis.T) )
            Adata[index:index+nEntries] = quadWeight * \
                np.ravel( np.outer(np.dot(gradphis, self.velocity), phis) )
            if not massLumping:
                Mdata[index:index+nEntries] = quadWeight * \
                    np.ravel( np.outer(phis, phis) )
            row_ind[index:index+nEntries] = np.repeat(inds, inds.size)
            col_ind[index:index+nEntries] = np.tile(inds, inds.size)
            index += nEntries
        
        del storage

        Kdata = np.concatenate((Kdata[:index], opMatBints[0]))
        Adata = np.concatenate((Adata[:index], opMatBints[1]))
        row_ind = np.concatenate((row_ind[:index], opMatBints[2]))
        col_ind = np.concatenate((col_ind[:index], opMatBints[3]))
        self.K = sp.csr_matrix( (Kdata, (row_ind, col_ind)),
                                shape=(nNodes, nNodes) )
        self.A = sp.csr_matrix( (Adata, (row_ind, col_ind)),
                                shape=(nNodes, nNodes) )
        if massLumping:
            self.M = sp.diags(self.u_weights, format='csr')
        else:
            self.M = sp.csr_matrix( (Mdata[:index], (row_ind[:index], col_ind[:index])),
                                shape=(nNodes, nNodes) )

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
        self.U = np.empty(self.X.size)
        for i, x in enumerate(self.X):
            self.U[i] = np.sum(self.phiPlot[i] * self.uPlot[self.indPlot[i]])

    def computePlottingSolution(self):
        """Compute interpolated solution at the plotting points.

        Returns
        -------
        None.

        """
        self.uPlot[0:self.nNodes] = self.uI
        for i, x in enumerate(self.X):
            self.U[i] = np.sum(self.phiPlot[i] * self.uPlot[self.indPlot[i]])
