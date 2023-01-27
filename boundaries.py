# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:46:03 2021

@author: Samuel A. Maloney

"""

from abc import ABCMeta, abstractmethod
from scipy.special import roots_legendre
import numpy as np
import scipy.sparse as sp
# import warnings


class Boundary(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    def __repr__(self):
        uniformSpacing = np.array((self.sim.xmax/self.sim.NX, self.sim.ymax/self.sim.NY))
        return f'{self.__class__.__name__}(support = {self.support/uniformSpacing})'

    def __init__(self, sim, support):
        self.sim = sim
        support = np.array(support).reshape(-1)
        uniformSpacing = np.array((sim.xmax/sim.NX, sim.ymax/sim.NY))
        if len(support) == 1:
            support = np.full(sim.ndim, support) * uniformSpacing
        elif len(support) != sim.ndim:
            raise TypeError('support must be scalar or array_like of length ndim')
        self.support = support
        self.rsupport = 1./support
        self.volume = (2.*(self.support + 0.5*uniformSpacing)).prod()

    @abstractmethod
    def computeNodes(self):
        raise NotImplementedError

    @abstractmethod
    def findNodesInSupport(self, p):
        raise NotImplementedError

    @abstractmethod
    def computeDisplacements(self, p, inds):
        raise NotImplementedError

    @abstractmethod
    def modifyOperatorMatrix(self, R, b):
        raise NotImplementedError

    @abstractmethod
    def computeBoundaryIntegrals(self, vci):
        raise NotImplementedError

    def w(self, p):
        indices, displacements, distances = self.findNodesInSupport(p)
        w = np.apply_along_axis(self.sim.kernel.w, 0, distances).prod(axis=1)
        return indices, w, displacements

    def dw(self, p):
        indices, displacements, distances = self.findNodesInSupport(p)
        w = np.empty((len(indices), self.sim.ndim))
        dwdr = np.empty(w.shape)
        for i in range(self.sim.ndim):
            w[:,i], dwdr[:,i] = self.sim.kernel.dw(distances[:,i])
        gradw = dwdr * np.sign(displacements) * self.rsupport
        gradw[:,0] *= w[:,1]
        gradw[:,1] *= w[:,0]
        w = np.prod(w, axis=1)
        return indices, w, gradw, displacements

    def d2w(self, p):
        raise NotImplementedError

    def mapping(self, points, zeta=0.):
        return self.sim.mapping(points, zeta)

    def __call__(self, p):
        return self.w(p)


class PeriodicBoundary(Boundary):
    @property
    def name(self):
        return 'periodic'

    def __init__(self, sim, support):
        super().__init__(sim, support)
        self.nNodes = sim.NX * sim.NY

    def computeNodes(self):
        self.nodes = np.vstack( (np.repeat(self.sim.nodeX[:-1], self.sim.NY),
                                self.sim.nodeY[:-1,:-1].ravel()) ).T
        return self.nodes

    def mapping(self, points, zeta=0.):
        # Note: negative numbers very close to zero (about -5e-9) may be
        # rounded to ymax after the 1st modulo, hence why the 2nd is needed.
        ymax = self.sim.ymax
        return self.sim.mapping(points, zeta) % ymax % ymax

    def findNodesInSupport(self, p):
        support = self.support
        xmax = self.sim.xmax
        ymax = self.sim.ymax
        indices = []
        displacements = []

        for iPlane in range(self.sim.NX):
            nodeX = self.sim.nodeX[iPlane]
            dispX = p.flat[0] - nodeX
            dispX = np.array((dispX, dispX + xmax, dispX - xmax))
            iX = np.argmin(np.abs(dispX))
            dispX = dispX[iX]
            if np.abs(dispX) > support[0]:
                continue
            maps = float(self.mapping(p.ravel(), nodeX))
            dispY = maps - self.sim.nodeY[iPlane][:-1]
            dispY = np.vstack((dispY, dispY + ymax, dispY - ymax))
            iY = np.abs(dispY).argmin(axis=0)
            dispY = dispY[(iY,np.arange(dispY.shape[1]))]
            inds = np.flatnonzero(np.abs(dispY) < support[1]).astype('uint32')
            dispY = dispY[inds].reshape(-1,1)
            indices.append(inds + iPlane*self.sim.NY)
            displacements.append(np.hstack((np.full((len(inds),1), dispX), dispY)))
        # combine and return results
        indices = np.concatenate(indices)
        displacements = np.concatenate(displacements) * self.rsupport
        distances = np.abs(displacements)
        return indices, displacements, distances

    def computeDisplacements(self, p, inds):
        xmax = self.sim.xmax
        ymax = self.sim.ymax
        disps = p.ravel() - self.nodes[inds]
        dispX = np.vstack((disps[:,0], disps[:,0] + xmax, disps[:,0] - xmax))
        dispY = np.vstack((disps[:,1], disps[:,1] + ymax, disps[:,1] - ymax))
        iX = np.abs(dispX).argmin(axis=0)
        iY = np.abs(dispY).argmin(axis=0)
        n = len(inds)
        disps[:,0] = dispX[(iX,np.arange(n))]
        disps[:,1] = dispY[(iY,np.arange(n))]
        return disps

    def modifyOperatorMatrix(self, R, b):
        return R, b

    def computeBoundaryIntegrals(self, vci):
        e = np.array(())
        if vci == 1:
            return np.zeros((self.nNodes, self.sim.ndim)), (e,e,e,e)
        elif vci == 2:
            return np.zeros((self.nNodes, self.sim.ndim, 3)), (e,e,e,e)


class DirichletBoundary(Boundary):
    @property
    def name(self):
        return 'Dirichlet'

    def __init__(self, sim, support, g, NDX=None):
        NX = sim.NX
        NY = sim.NY
        dx = sim.dx
        nodeX = sim.nodeX
        self.NDX = NDX
        self.g = g
        self.B = sim.mapping.B
        super().__init__(sim, support)
        if (NDX is None) or (NDX == 1):
            self.topNodeX = np.array(())
            self.bottomNodeX = self.topNodeX
        elif isinstance(NDX, int):
            if NDX < 2:
                raise ValueError('NDX must be at least 2')
            self.topNodeX = np.concatenate( [
                np.arange(nodeX[i] + dx[i]/NDX, nodeX[i+1] - 0.5*dx[i]/NDX, dx[i]/NDX)
                for i in range(NX) ] )
            self.bottomNodeX = self.topNodeX
        self.nBoundaryNodes = 2*(NY + NX) + \
            self.topNodeX.size + self.bottomNodeX.size
        self.nNodes = self.nBoundaryNodes + (NX - 1)*(NY - 1)

    def computeNodes(self):
        NXY = (self.sim.NX + 1) * (self.sim.NY + 1)
        nodeY = self.sim.nodeY
        nodes = np.empty((self.nNodes, 2))
        nodes[:NXY] = np.vstack((
            np.repeat(self.sim.nodeX, self.sim.NY + 1), nodeY.ravel() )).T
        if self.NDX is not None:
            # bottom boundary
            nodes[NXY:NXY+self.bottomNodeX.size:,0] = self.bottomNodeX
            nodes[NXY:NXY+self.bottomNodeX.size:,1] = 0.
            # top boundary
            nodes[-self.topNodeX.size:,0] = self.topNodeX
            nodes[-self.topNodeX.size:,1] = self.sim.ymax
        self.isBoundaryNode = ((nodes == 0.) | (nodes == 1.)).any(axis=1)
        self.nodes = nodes
        return nodes

    def findNodesInSupport(self, p):
        NX = self.sim.NX
        NY = self.sim.NY
        NXY = (NX + 1)*(NY + 1)
        support = self.support
        indices = []
        displacements = []
        # check nodes on FCI planes
        for iPlane in range(NX+1):
            nodeX = self.sim.nodeX[iPlane]
            dispX = p.flat[0] - nodeX
            if np.abs(dispX) > support[0]:
                continue
            maps = float(self.mapping(p.ravel(), nodeX))
            dispY = maps - self.sim.nodeY[iPlane]
            inds = np.flatnonzero(np.abs(dispY) < support[1]).astype('uint32')
            dispY = dispY[inds].reshape(-1,1)
            indices.append(inds + iPlane*(NY + 1))
            displacements.append(np.hstack((np.full((len(inds),1), dispX), dispY)))
        # check additional nodes on bottom boundary
        dispX = p.flat[0] - self.bottomNodeX
        inds = np.flatnonzero(np.abs(dispX) < support[0]).astype('uint32')
        for i in inds:
            dispY = float(self.mapping(p.ravel(), self.bottomNodeX[i])) - 0.
            if np.abs(dispY) < support[1]:
                indices.append(np.array([i + NXY]))
                displacements.append(np.array([[dispX[i], dispY]]))
        # check additional nodes on top boundary
        dispX = p.flat[0] - self.topNodeX
        inds = np.flatnonzero(np.abs(dispX) < support[0]).astype('uint32')
        for i in inds:
            dispY = float(self.mapping(p.ravel(), self.topNodeX[i])) - self.sim.ymax
            if np.abs(dispY) < support[1]:
                indices.append(np.array([i + NXY + self.bottomNodeX.size]))
                displacements.append(np.array([[dispX[i], dispY]]))
        # combine and return results
        indices = np.concatenate(indices)
        displacements = np.concatenate(displacements) * self.rsupport
        distances = np.abs(displacements)
        return indices, displacements, distances

    def computeDisplacements(self, p, inds):
        return p.ravel() - self.nodes[inds]

    def modifyOperatorMatrix(self, R, b):
        # Apply BCs using Lagrange multipliers
        nMaxEntries = int((self.nNodes * self.volume)**2 * self.nBoundaryNodes)
        data = np.zeros(nMaxEntries)
        row_ind = np.zeros(nMaxEntries, dtype='uint32')
        col_ind = np.zeros(nMaxEntries, dtype='uint32')
        index = 0
        boundaryNodes = self.nodes[self.isBoundaryNode]
        for iN, node in enumerate(boundaryNodes):
            indices, phis = self.sim.phi(node)
            nEntries = len(indices)
            data[index:index+nEntries] = phis
            row_ind[index:index+nEntries] = indices
            col_ind[index:index+nEntries] = np.repeat(iN, nEntries)
            index += nEntries
        inds = np.flatnonzero(data.round(decimals=14,out=data))
        G = sp.csr_matrix( (data[inds], (row_ind[inds], col_ind[inds])),
                            shape=(self.nNodes, self.nBoundaryNodes) )
        boundaryValues = self.g(boundaryNodes)
        modR = sp.bmat([[R, G], [G.T, None]], format='csr')
        modb = np.concatenate((b, boundaryValues))
        return modR, modb

    def computeBoundaryIntegrals(self, vci):
        ndim = self.sim.ndim
        nNodes = self.nNodes
        NX = self.sim.NX
        quadType = self.sim.quadType
        Qord = self.sim.Qord
        NQX = self.sim.NQX
        NQY = self.sim.NQY
        xmax = self.sim.xmax
        ymax = self.sim.ymax
        diffusivity = self.sim.diffusivity
        velocity = self.sim.velocity
        # pre-allocate arrays for operator matrix triplets
        nQuads = 2*Qord*(NX*NQX + NQY)
        nMaxEntries = int((nNodes * self.volume)**2 * nQuads)
        Kdata = np.empty(nMaxEntries)
        Adata = np.empty(nMaxEntries)
        row_ind = np.empty(nMaxEntries, dtype='int')
        col_ind = np.empty(nMaxEntries, dtype='int')
        index = 0
        
        if vci == 1:
            integrals = np.zeros((nNodes, ndim))
            phiSums = integrals
        elif vci == 2:
            integrals = np.zeros((nNodes, ndim, 3))
            phiSums = integrals[:,:,0]
        if quadType.lower() in ('gauss', 'g', 'gaussian'):
            offsets, weights = roots_legendre(Qord)
        elif quadType.lower() in ('uniform', 'u'):
            offsets = np.arange(1/Qord - 1, 1, 2/Qord)
            weights = np.full(Qord, 2/Qord)
        # Left/Right boundaries
        yfac = 0.5 * ymax / NQY
        quads = np.arange(yfac, ymax, 2*yfac)
        quadWeights = np.tile(yfac * weights, len(quads))
        quads = (np.repeat(quads, Qord).reshape(-1,Qord) + yfac*offsets).ravel()
        for iQ, quad in enumerate(quads):
            quadWeight = quadWeights[iQ]
            # Left boundary
            inds, phis, gradphis = self.sim.dphi(np.array((0, quad)))
            # Operator matrix contributions
            nEntries = len(inds)**2
            Kdata[index:index+nEntries] = quadWeight * \
                np.ravel( np.outer(phis, diffusivity[0] @ gradphis.T) )
            Adata[index:index+nEntries] = quadWeight * \
                np.ravel( np.outer(phis,  velocity[0] * phis) )
            row_ind[index:index+nEntries] = np.repeat(inds, len(inds))
            col_ind[index:index+nEntries] = np.tile(inds, len(inds))
            index += nEntries
            # VCI residuals
            phiSums[inds,0] -= phis * quadWeight
            if vci == 2:
                disps = self.computeDisplacements(np.array((0, quad)), inds)
                integrals[inds,0,1:] -= phis[:,np.newaxis] * disps * quadWeight
            # Right boundary
            inds, phis, gradphis = self.sim.dphi(np.array((xmax, quad)))
            # Operator matrix contributions
            nEntries = len(inds)**2
            Kdata[index:index+nEntries] = quadWeight * \
                np.ravel( np.outer(phis, -diffusivity[0] @ gradphis.T) )
            Adata[index:index+nEntries] = quadWeight * \
                np.ravel( np.outer(phis,  -velocity[0] * phis) )
            row_ind[index:index+nEntries] = np.repeat(inds, len(inds))
            col_ind[index:index+nEntries] = np.tile(inds, len(inds))
            index += nEntries
            # VCI residuals
            phiSums[inds,0] += phis * quadWeight
            if vci == 2:
                disps = self.computeDisplacements(np.array((xmax, quad)), inds)
                integrals[inds,0,1:] += phis[:,np.newaxis] * disps * quadWeight
        # Top/Bottom boundaries
        xfac = 0.5 / NQX
        for iPlane in range(NX):
            dx = self.sim.dx[iPlane]
            quads = dx*(0.5 + xfac*offsets) + self.sim.nodeX[iPlane]
            quadWeights = dx * xfac * weights
            for iQ, quad in enumerate(quads):
                quadWeight = quadWeights[iQ]
                # Bottom boundary
                inds, phis, gradphis = self.sim.dphi(np.array((quad, 0)))
                # Operator matrix contributions
                nEntries = len(inds)**2
                Kdata[index:index+nEntries] = quadWeight * \
                    np.ravel( np.outer(phis, diffusivity[1] @ gradphis.T) )
                Adata[index:index+nEntries] = quadWeight * \
                    np.ravel( np.outer(phis,  velocity[1] * phis) )
                row_ind[index:index+nEntries] = np.repeat(inds, len(inds))
                col_ind[index:index+nEntries] = np.tile(inds, len(inds))
                index += nEntries
                # VCI residuals
                phiSums[inds,1] -= phis * quadWeight
                if vci == 2:
                    disps = self.computeDisplacements(np.array((quad, 0)), inds)
                    integrals[inds,1,1:] -= phis[:,np.newaxis] * disps * quadWeight
                # Top boundary
                inds, phis, gradphis = self.sim.dphi(np.array((quad, ymax)))
                # Operator matrix contributions
                nEntries = len(inds)**2
                Kdata[index:index+nEntries] = quadWeight * \
                    np.ravel( np.outer(phis, -diffusivity[1] @ gradphis.T) )
                Adata[index:index+nEntries] = quadWeight * \
                    np.ravel( np.outer(phis,  -velocity[1] * phis) )
                row_ind[index:index+nEntries] = np.repeat(inds, len(inds))
                col_ind[index:index+nEntries] = np.tile(inds, len(inds))
                index += nEntries
                # VCI residuals
                phiSums[inds,1] += phis * quadWeight
                if vci == 2:
                    disps = self.computeDisplacements(np.array((quad, ymax)), inds)
                    integrals[inds,1,1:] += phis[:,np.newaxis] * disps * quadWeight
        return integrals, (Kdata[:index], Adata[:index], row_ind[:index], col_ind[:index])


# class DirichletXPeriodicYBoundary(Boundary):
#     @property
#     def name(self):
#         return 'DirichletXPeriodicY'

#     def __init__(self, sim, g):
#         self.nXnodes = sim.NX - 1
#         self.nYnodes = sim.NY
#         super().__init__(sim)
#         self.g = g
#         self.nDirichletNodes = 2*self.nYnodes
#         self.nNodes = self.nNodes + self.nDirichletNodes

#     def computeNodes(self):
#         nNodes = self.nNodes
#         nYnodes = self.nYnodes
#         nodeY = self.sim.nodeY
#         self.nodes = np.empty((self.nNodes, 2))
#         self.nodes[:nNodes] = np.vstack((
#             np.repeat(self.sim.nodeX[1:-1], self.sim.NY),
#             nodeY[1:-1,:-1].ravel() )).T
#         # left boundary
#         self.nodes[-nYnodes:] = np.vstack((
#             np.zeros(nYnodes), nodeY[0][-2::-1] )).T
#         # right boundary
#         self.nodes[-2*nYnodes:-nYnodes] = np.vstack((
#             np.full(nYnodes, self.sim.xmax), nodeY[-1][-2::-1] )).T
#         return self.nodes

#     def mapping(self, points, zeta=0.):
#         # Note: negative numbers very close to zero (about -3.5e-10) may be
#         # rounded to 1.0 after the 1st modulo, hence why the 2nd is needed.
#         return self.sim.mapping(points, zeta) % 1 % 1
