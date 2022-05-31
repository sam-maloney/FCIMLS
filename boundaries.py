# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:46:03 2021

@author: Samuel A. Maloney

"""

from abc import ABCMeta, abstractmethod
import numpy as np
import warnings


class Boundary(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    def __init__(self, sim, support):
        self.sim = sim
        self.nDoFs = self.nXnodes * self.nYnodes
        support = np.array(support).reshape(-1)
        uniformSpacing = np.array((sim.xmax/sim.NX, sim.ymax/sim.NY))
        if len(support) == 1:
            support = np.full(sim.ndim, support) * uniformSpacing
        elif len(support) != sim.ndim:
            raise TypeError('support must be scalar or array_like of length ndim')
        self.ndim = sim.ndim
        self.support = support
        self.rsupport = 1./support
        self.volume = (2.*(self.support + 0.5*uniformSpacing)).prod()

    @abstractmethod
    def w(self, p):
        raise NotImplementedError

    @abstractmethod
    def dw(self, p):
        raise NotImplementedError

    @abstractmethod
    def d2w(self, p):
        raise NotImplementedError

    @abstractmethod
    def computeNodes(self):
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
        self.nXnodes = sim.NX
        self.nYnodes = sim.NY
        super().__init__(sim, support)
        self.nNodes = self.nDoFs

    def computeNodes(self):
        self.DoFs = np.vstack( (np.repeat(self.sim.nodeX[:-1], self.sim.NY),
                                self.sim.nodeY[:-1,:-1].ravel()) ).T
        self.nodes = self.DoFs
        return self.DoFs

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
            disps = maps - self.sim.nodeY[iPlane][:-1]
            disps = np.vstack((disps, disps + ymax, disps - ymax))
            iY = np.abs(disps).argmin(axis=0)
            disps = disps[(iY,np.arange(disps.shape[1]))]
            inds = np.flatnonzero(np.abs(disps) < support[1]).astype('uint32')
            disps = disps[inds].reshape(-1,1)
            indices.append(inds + iPlane*self.sim.NY)
            displacements.append(np.hstack((np.full((len(inds),1), dispX), disps)))

        indices = np.concatenate(indices)
        displacements = np.concatenate(displacements) * self.rsupport
        distances = np.abs(displacements)
        return indices, displacements, distances

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


# class DirichletBoundary(Boundary):
#     @property
#     def name(self):
#         return 'Dirichlet'

#     def __init__(self, sim, support, g, NDX=None):
#         NX = sim.NX
#         nodeX = sim.nodeX
#         self.NDX = NDX
#         self.nXnodes = NX - 1
#         self.nYnodes = sim.NY - 1
#         super().__init__(sim, support)
#         self.g = g
#         self.B = sim.mapping.B
#         # gradphis[ind, x=0/y=1, dphi=0/drho=1]
#         self.gradphis = np.empty((4,2,2))
#         if NDX is None:
#             self.DirichletNodeX = [sim.nodeX, sim.nodeX]
#         elif (type(NDX) is int) and (NDX > 0):
#             self.DirichletNodeX = np.empty(NX*NDX+1)
#             self.DirichletNodeX[-1] = self.sim.xmax
#             self.DirichletNodeX[:-1] = np.concatenate(
#                 [ np.linspace(nodeX[i], nodeX[i+1], NDX, False)
#                   for i in range(NX) ] )
#             self.DirichletNodeX = [self.DirichletNodeX, self.DirichletNodeX]
#         elif (type(NDX) is int) and (NDX < 0):
#             NDX = -NDX

#             ##### map nearest y-nodes, fill full space with max NDX divisions #####
#             bottomNodes = []
#             topNodes = []
#             # ignore warnings about nan's where p doesn't map to any boundaries
#             with warnings.catch_warnings():
#                 warnings.filterwarnings('ignore', r'invalid value encountered')
#                 for iPlane in range(NX):
#                     dx = sim.dx[iPlane]
#                     zetaL = nodeX[iPlane]
#                     zetaR = nodeX[iPlane+1]
#                     zetaBL, zetaTL = map(float,
#                         self.B(np.array((zetaL, sim.nodeY[iPlane,1]))) )
#                     zetaBR, zetaTR = map(float,
#                         self.B(np.array((zetaR, sim.nodeY[iPlane+1,1]))) )
#                     if (zetaBL > zetaL) and (zetaBL < zetaR):
#                         dzeta = max(zetaBL - zetaL, dx/NDX)
#                         num = int((dx + 0.1*dzeta) / dzeta)
#                         bottomNodes.append(np.linspace(zetaL, zetaR, num, False))
#                     elif (zetaBR > zetaL) and (zetaBR < zetaR):
#                         dzeta = max(zetaR - zetaBR, dx/NDX)
#                         num = int((dx + 0.1*dzeta) / dzeta)
#                         bottomNodes.append(np.linspace(zetaL, zetaR, num, False))
#                     else:
#                         bottomNodes.append(np.array([zetaL]))
#                     zetaBL, zetaTL = map(float,
#                         self.B(np.array((zetaL, sim.nodeY[iPlane,-2]))) )
#                     zetaBR, zetaTR = map(float,
#                         self.B(np.array((zetaR, sim.nodeY[iPlane+1,-2]))) )
#                     if (zetaTL > zetaL) and (zetaTL < zetaR):
#                         dzeta = max(zetaTL - zetaL, dx/NDX)
#                         num = int((dx + 0.1*dzeta) / dzeta)
#                         topNodes.append(np.linspace(zetaL, zetaR, num, False))
#                     elif (zetaTR > zetaL) and (zetaTR < zetaR):
#                         dzeta = max(zetaR - zetaTR, dx/NDX)
#                         num = int((dx + 0.1*dzeta) / dzeta)
#                         topNodes.append(np.linspace(zetaL, zetaR, num, False))
#                     else:
#                         topNodes.append(np.array([zetaL]))
#             bottomNodes.append(np.array([self.sim.xmax]))
#             topNodes.append(np.array([self.sim.xmax]))
#             self.DirichletNodeX = [np.concatenate(bottomNodes),
#                                     np.concatenate(topNodes)]

#             # ##### map nearest y-nodes, fill remaining space with max NDX-1 divisions #####
#             # bottomNodes = []
#             # topNodes = []
#             # # ignore warnings about nan's where p doesn't map to any boundaries
#             # with warnings.catch_warnings():
#             #     warnings.filterwarnings('ignore', r'invalid value encountered')
#             #     for iPlane in range(NX):
#             #         zetaL = nodeX[iPlane]
#             #         zetaR = nodeX[iPlane+1]
#             #         zetaBL, zetaTL = map(float,
#             #             self.B(np.array((zetaL, sim.nodeY[iPlane,1]))) )
#             #         zetaBR, zetaTR = map(float,
#             #             self.B(np.array((zetaR, sim.nodeY[iPlane+1,1]))) )
#             #         if (zetaBL > zetaL) and (zetaBL < zetaR):
#             #             dx = max(zetaBL - zetaL, (zetaR - zetaBL)/(NDX-1))
#             #             num = int((zetaR - zetaBL + 0.1*dx) / dx)
#             #             bottomNodes.append(np.concatenate((np.array([zetaL]),
#             #                 np.linspace(zetaBL, zetaR, num, False))))
#             #         elif (zetaBR > zetaL) and (zetaBR < zetaR):
#             #             dx = max(zetaR - zetaBR, (zetaBR - zetaL)/(NDX-1))
#             #             num = int((zetaBR - zetaL + 0.1*dx) / dx) + 1
#             #             bottomNodes.append(np.linspace(zetaL, zetaBR, num))
#             #         else:
#             #             bottomNodes.append(np.array([zetaL]))
#             #         zetaBL, zetaTL = map(float,
#             #             self.B(np.array((zetaL, sim.nodeY[iPlane,-2]))) )
#             #         zetaBR, zetaTR = map(float,
#             #             self.B(np.array((zetaR, sim.nodeY[iPlane+1,-2]))) )
#             #         if (zetaTL > zetaL) and (zetaTL < zetaR):
#             #             dx = max(zetaTL - zetaL, (zetaR - zetaTL)/(NDX-1))
#             #             num = int((zetaR - zetaTL + 0.1*dx) / dx)
#             #             topNodes.append(np.concatenate((np.array([zetaL]),
#             #                 np.linspace(zetaTL, zetaR, num, False))))
#             #         elif (zetaTR > zetaL) and (zetaTR < zetaR):
#             #             dx = max(zetaR - zetaTR, (zetaTR - zetaL)/(NDX-1))
#             #             num = int((zetaTR - zetaL + 0.1*dx) / dx) + 1
#             #             topNodes.append(np.linspace(zetaL, zetaTR, num))
#             #         else:
#             #             topNodes.append(np.array([zetaL]))
#             # bottomNodes.append(np.array([self.sim.xmax]))
#             # topNodes.append(np.array([self.sim.xmax]))
#             # self.DirichletNodeX = [np.concatenate(bottomNodes),
#             #                         np.concatenate(topNodes)]

#             # ##### map nearest y-nodes, fill remaining space with NDX-1 divisions #####
#             # bottomNodes = -np.ones(NX*NDX+1)
#             # topNodes = -np.ones(NX*NDX+1)
#             # # ignore warnings about nan's where p doesn't map to any boundaries
#             # with warnings.catch_warnings():
#             #     warnings.filterwarnings('ignore', r'invalid value encountered')
#             #     for iPlane in range(NX):
#             #         zetaL = nodeX[iPlane]
#             #         zetaR = nodeX[iPlane+1]
#             #         for y in (sim.nodeY[iPlane,1], sim.nodeY[iPlane,-2]):
#             #             zetaBottom, zetaTop = self.B(np.array((zetaL, y)))
#             #             zetaBottom = float(zetaBottom)
#             #             zetaTop = float(zetaTop)
#             #             if (zetaBottom > zetaL) and (zetaBottom < zetaR):
#             #                 bottomNodes[iPlane*NDX] = nodeX[iPlane]
#             #                 bottomNodes[iPlane*NDX+1:(iPlane+1)*NDX] = \
#             #                     np.linspace(zetaBottom, zetaR, NDX-1, False)
#             #             if (zetaTop > zetaL) and (zetaTop < zetaR):
#             #                 topNodes[iPlane*NDX] = nodeX[iPlane]
#             #                 topNodes[iPlane*NDX+1:(iPlane+1)*NDX] = \
#             #                     np.linspace(zetaTop, zetaR, NDX-1, False)
#             #         for y in (sim.nodeY[iPlane+1,1], sim.nodeY[iPlane+1,-2]):
#             #             zetaBottom, zetaTop = self.B(np.array((zetaR, y)))
#             #             zetaBottom = float(zetaBottom)
#             #             zetaTop = float(zetaTop)
#             #             if (zetaBottom > zetaL) and (zetaBottom < zetaR):
#             #                 bottomNodes[iPlane*NDX:(iPlane+1)*NDX] = \
#             #                     np.linspace(zetaL, zetaBottom, NDX, True)
#             #             if (zetaTop > zetaL) and (zetaTop < zetaR):
#             #                 topNodes[iPlane*NDX:(iPlane+1)*NDX] = \
#             #                     np.linspace(zetaL, zetaTop, NDX, True)
#             # bottomNodes[-1] = self.sim.xmax
#             # topNodes[-1] = self.sim.xmax
#             # if (np.any(topNodes == -1.0) or np.any(bottomNodes == -1.0)):
#             #     print('Error generating Dirichlet nodes')
#             # self.DirichletNodeX = [bottomNodes, topNodes]
#         else: # autogenerate top/bottom boundary nodes from nodeX/nodeY
#             bottomNodes = []
#             topNodes = []
#             # ignore warnings about nan's where p doesn't map to any boundaries
#             with warnings.catch_warnings():
#                 warnings.filterwarnings('ignore', r'invalid value encountered')
#                 for iPlane in range(NX):
#                     zetaL = nodeX[iPlane]
#                     zetaR = nodeX[iPlane+1]
#                     bottomNodes.append(zetaL)
#                     topNodes.append(zetaL)
#                     for y in sim.nodeY[iPlane,1:-1]:
#                         zetaBottom, zetaTop = self.B(np.array((zetaL, y)))
#                         zetaBottom = float(zetaBottom)
#                         zetaTop = float(zetaTop)
#                         if (zetaBottom > zetaL) and (zetaBottom < zetaR):
#                             bottomNodes.append(zetaBottom)
#                         if (zetaTop > zetaL) and (zetaTop < zetaR):
#                             topNodes.append(zetaTop)
#                     for y in sim.nodeY[iPlane+1,1:-1]:
#                         zetaBottom, zetaTop = self.B(np.array((zetaR, y)))
#                         zetaBottom = float(zetaBottom)
#                         zetaTop = float(zetaTop)
#                         if (zetaBottom > zetaL) and (zetaBottom < zetaR):
#                             bottomNodes.append(zetaBottom)
#                         if (zetaTop > zetaL) and (zetaTop < zetaR):
#                             topNodes.append(zetaTop)
#             bottomNodes.append(self.sim.xmax)
#             topNodes.append(self.sim.xmax)
#             self.DirichletNodeX = [np.sort(bottomNodes), np.sort(topNodes)]

#         self.nDirichletNodes = 2*self.nYnodes + self.DirichletNodeX[0].size \
#                                               + self.DirichletNodeX[1].size
#         self.nNodes = self.nDoFs + self.nDirichletNodes

#     def computeNodes(self):
#         nDoFs = self.nDoFs
#         nYnodes = self.nYnodes
#         nodeY = self.sim.nodeY
#         self.nodes = np.empty((self.nNodes, 2))
#         self.nodes[:nDoFs] = np.vstack((
#             np.repeat(self.sim.nodeX[1:-1], self.sim.NY-1),
#             nodeY[1:-1,1:-1].ravel() )).T
#         # left boundary
#         self.nodes[-nYnodes:] = np.vstack((
#             np.zeros(nYnodes), nodeY[0][-2:0:-1] )).T
#         # right boundary
#         self.nodes[-2*nYnodes:-nYnodes] = np.vstack((
#             np.full(nYnodes, self.sim.xmax), nodeY[-1][-2:0:-1] )).T
#         # bottom boundary
#         nBottomNodes = self.DirichletNodeX[0].size
#         self.nodes[-2*nYnodes - nBottomNodes:-2*nYnodes,0] = \
#             self.DirichletNodeX[0][-1::-1]
#         self.nodes[-2*nYnodes - nBottomNodes:-2*nYnodes,1] = 0.
#         # top boundary
#         self.nodes[nDoFs:-2*nYnodes - nBottomNodes,0] = \
#             self.DirichletNodeX[1][-1::-1]
#         self.nodes[nDoFs:-2*nYnodes - nBottomNodes,1] = 1.
#         return self.nodes

#     def __call__(self, p, iPlane=None):
#         if iPlane is None:
#             iPlane = np.searchsorted(self.sim.nodeX[1:], p[0])
#         originalShape = p.shape
#         p.shape = (2,)
#         nXnodes = self.nXnodes
#         nYnodes = self.nYnodes
#         nBottomNodes = self.DirichletNodeX[0].size
#         nodeX = self.sim.nodeX[iPlane]
#         nodeXp1 = self.sim.nodeX[iPlane + 1]
#         nodeY = self.sim.nodeY[iPlane]
#         nodeYp1 = self.sim.nodeY[iPlane + 1]
#         isBoundaryMinus = isBoundaryPlus = False
#         phis = np.empty(4)
#         self.gradphis.fill(0.0)
#         gradRho = 1.0 / (nodeXp1 - nodeX)
#         self.gradphis[:,0,1] = np.array((-gradRho, -gradRho, gradRho, gradRho))

#         # ignore warnings about nan's where p doesn't map to any boundaries
#         with warnings.catch_warnings():
#             warnings.filterwarnings('ignore', r'invalid value encountered')
#             zetaBottom, zetaTop = self.B(p)
#             zetaBottom = float(zetaBottom)
#             zetaTop = float(zetaTop)
#             isBoundaryBottom = (zetaBottom > nodeX) * (zetaBottom < nodeXp1)
#             isBoundaryTop = (zetaTop > nodeX) * (zetaTop < nodeXp1)

#             if isBoundaryBottom and (zetaBottom <= p[0]):
#                 zetaMinus = zetaBottom
#                 isBoundaryMinus = True
#                 dBdx, dBdy = self.sim.mapping.dBbottom(p)
#             if isBoundaryTop and (zetaTop <= p[0]):
#                 zetaMinus = zetaTop
#                 isBoundaryMinus = True
#                 dBdx, dBdy = self.sim.mapping.dBtop(p)
#             if isBoundaryMinus:
#                 DirichletNodeX = self.DirichletNodeX[int(isBoundaryTop)]
#                 iR = np.searchsorted(DirichletNodeX, zetaMinus, side='right')
#                 i0 = -2*nYnodes - nBottomNodes*int(isBoundaryTop) - iR - 1
#                 i1 = i0 + 1
#                 zeta0 = DirichletNodeX[iR]
#                 zeta1 = DirichletNodeX[iR-1]
#                 idx = 1.0 / (zeta0 - zeta1)
#                 self.gradphis[0,0,0] = dBdx * idx
#                 self.gradphis[1,0,0] = -self.gradphis[0,0,0]
#                 self.gradphis[2:,0,1] = \
#                     (nodeXp1 - zetaMinus + dBdx*(p[0] - nodeXp1)) / \
#                     (zetaMinus - nodeXp1)**2
#                 self.gradphis[:2,0,1] = -self.gradphis[2:,0,1]

#                 self.gradphis[0,1,0] = dBdy * idx
#                 self.gradphis[1,1,0] = -self.gradphis[0,1,0]
#                 self.gradphis[2:,1,1] = dBdy*(p[0] - nodeXp1) \
#                                       / (zetaMinus - nodeXp1)**2
#                 self.gradphis[:2,1,1] = -self.gradphis[2:,1,1]
#                 phis[0] = (zetaMinus - zeta1) * idx
#                 phis[1] = 1 - phis[0]
#                 g0 = self.g(np.array((zeta0, float(isBoundaryTop))))
#                 phis[0] *= g0
#                 self.gradphis[0,:,0] *= g0
#                 g1 = self.g(np.array((zeta1, float(isBoundaryTop))))
#                 phis[1] *= g1
#                 self.gradphis[1,:,0] *= g1
#                 # if not np.allclose(np.array((zeta0, float(isBoundaryTop))), self.sim.nodes[i0]):
#                 #     print(f'p0 = ({zeta0}, {float(isBoundaryTop)}),\t node = {self.sim.nodes[i0]})')
#                 # if not np.allclose(np.array((zeta1, float(isBoundaryTop))), self.sim.nodes[i1]):
#                 #     print(f'p1 = ({zeta1}, {float(isBoundaryTop)}),\t node = {self.sim.nodes[i1]})')

#             if isBoundaryBottom and (zetaBottom > p[0]):
#                 zetaPlus = zetaBottom
#                 isBoundaryPlus = True
#                 dBdx, dBdy = self.sim.mapping.dBbottom(p)
#             if isBoundaryTop and (zetaTop > p[0]):
#                 zetaPlus = zetaTop
#                 isBoundaryPlus = True
#                 dBdx, dBdy = self.sim.mapping.dBtop(p)
#             if isBoundaryPlus:
#                 DirichletNodeX = self.DirichletNodeX[int(isBoundaryTop)]
#                 iR = np.searchsorted(DirichletNodeX, zetaPlus, side='right')
#                 i2 = -2*nYnodes - nBottomNodes*int(isBoundaryTop) - iR - 1
#                 i3 = i2 + 1
#                 zeta2 = DirichletNodeX[iR]
#                 zeta3 = DirichletNodeX[iR-1]
#                 idx = 1.0 / (zeta2 - zeta3)
#                 self.gradphis[2,0,0] = dBdx * idx
#                 self.gradphis[3,0,0] = -self.gradphis[2,0,0]
#                 self.gradphis[:2,0,1] = \
#                     (nodeX - zetaPlus + dBdx*(p[0] - nodeX)) / \
#                     (zetaPlus - nodeX)**2
#                 self.gradphis[2:,0,1] = -self.gradphis[:2,0,1]

#                 self.gradphis[2,1,0] = dBdy * idx
#                 self.gradphis[3,1,0] = -self.gradphis[2,1,0]
#                 self.gradphis[:2,1,1] = dBdy*(p[0] - nodeX) \
#                                       / (zetaPlus - nodeX)**2
#                 self.gradphis[2:,1,1] = -self.gradphis[:2,1,1]

#                 phis[2] = (zetaPlus - zeta3) * idx
#                 phis[3] = 1 - phis[2]
#                 g2 = self.g(np.array((zeta2, float(isBoundaryTop))))
#                 phis[2] *= g2
#                 self.gradphis[2,:,0] *= g2
#                 g3 = self.g(np.array((zeta3, float(isBoundaryTop))))
#                 phis[3] *= g3
#                 self.gradphis[3,:,0] *= g3
#                 # if not np.allclose(np.array((zeta2, float(isBoundaryTop))), self.sim.nodes[i2]):
#                 #     print(f'p2 = ({zeta2}, {float(isBoundaryTop)}),\t node = {self.sim.nodes[i2]})')
#                 # if not np.allclose(np.array((zeta3, float(isBoundaryTop))), self.sim.nodes[i3]):
#                 #     print(f'p3 = ({zeta3}, {float(isBoundaryTop)}),\t node = {self.sim.nodes[i3]})')

#         if not isBoundaryMinus:
#             zetaMinus = nodeX
#             mapL = float(self.mapping(p, nodeX))
#             # if (mapL < 0.0) or (mapL > 1.0):
#             #     print('mapping out of range')
#             i1 = np.searchsorted(nodeY, mapL, side='right') - 1
#             if i1 > nYnodes: # for points right at top boundary nodes
#                 i1 -= 1
#             if i1 < 0:
#                 i1 = 0
#             i0 = i1 - 1
#             phis[1] = (mapL - nodeY[i1]) * self.sim.idy[iPlane][i1]
#             phis[0] = 1 - phis[1]
#             self.gradphis[1,1,0] = self.sim.idy[iPlane][i1]
#             self.gradphis[0,1,0] = -self.gradphis[1,1,0]
#             self.gradphis[:2,0,0] = -self.sim.mapping.deriv(p)*self.gradphis[:2,1,0]
#             ##### if i0 on the left or lower boundary #####
#             if (iPlane == 0) or (i0 < 0):
#                 g0 = self.g(np.array((nodeX, nodeY[i1])))
#                 phis[0] *= g0
#                 self.gradphis[0,:,0] *= g0
#                 if i0 < 0:
#                     i0 = -2*nYnodes - 1 - np.argwhere(
#                         abs(self.DirichletNodeX[0] - nodeX) < 1e-10)[0,0]
#                 elif iPlane == 0:
#                     i0 = -i0 - 1
#                 # if not np.allclose(np.array((nodeX, nodeY[i1])), self.sim.nodes[i0]):
#                 #     print(f'p0 = ({nodeX}, {nodeY[i1]}),\t node = {self.sim.nodes[i0]})')
#             else: # i0 is interior
#                 i0 += (iPlane - 1)*nYnodes
#             ##### if i1 on the left or top boundary #####
#             if (iPlane == 0) or (i1 >= nYnodes):
#                 # p1 = np.array((nodeX, nodeY[i1+1]))
#                 g1 = self.g(np.array((nodeX, nodeY[i1+1])))
#                 phis[1] *= g1
#                 self.gradphis[1,:,0] *= g1
#                 if i1 >= nYnodes:
#                     i1 = -2*nYnodes - nBottomNodes - 1 - np.argwhere(
#                         abs(self.DirichletNodeX[1] - nodeX) < 1e-10)[0,0]
#                 elif iPlane == 0:
#                     i1 = -i1 - 1
#                 # if not np.allclose(p1, self.sim.nodes[i1]):
#                 #     print(f'p1 = ({p1[0]}, {p1[1]}),\t node = {self.sim.nodes[i1]})')
#             else: # i1 is interior
#                 i1 += (iPlane - 1)*nYnodes
#         if not isBoundaryPlus:
#             zetaPlus = nodeXp1
#             mapR = float(self.mapping(p, nodeXp1))
#             # if (mapR < 0.0) or (mapR > 1.0):
#             #     print('mapping out of range')
#             i3 = np.searchsorted(nodeYp1, mapR, side='right') - 1
#             if i3 > nYnodes: # for points right at top boundary nodes
#                 i3 -= 1
#             if i3 < 0:
#                 i3 = 0
#             i2 = i3 - 1
#             phis[3] = (mapR - nodeYp1[i3]) * self.sim.idy[iPlane+1][i3]
#             phis[2] = 1 - phis[3]
#             self.gradphis[3,1,0] = self.sim.idy[iPlane+1][i3]
#             self.gradphis[2,1,0] = -self.gradphis[3,1,0]
#             self.gradphis[2:,0,0] = -self.sim.mapping.deriv(p)*self.gradphis[2:,1,0]
#             ##### if i2 on the right or lower boundary #####
#             if (iPlane == nXnodes) or (i2 < 0):
#                 g2 = self.g(np.array((nodeXp1, nodeYp1[i3])))
#                 phis[2] *= g2
#                 self.gradphis[2,:,0] *= g2
#                 if i2 < 0:
#                     i2 = -2*nYnodes - 1 - np.argwhere(
#                         abs(self.DirichletNodeX[0] - nodeXp1) < 1e-10)[0,0]
#                 elif iPlane == nXnodes:
#                     i2 = -i2 - nYnodes - 1
#                 # if not np.allclose(np.array((nodeXp1, nodeYp1[i3])), self.sim.nodes[i2]):
#                 #     print(f'p2 = ({nodeXp1}, {nodeYp1[i3]}),\t node = {self.sim.nodes[i2]})')
#             else:
#                 i2 += iPlane*nYnodes
#             ##### if i3 on the right or top boundary #####
#             if (iPlane == nXnodes) or (i3 >= nYnodes):
#                 # p3 = np.array((nodeXp1, nodeYp1[i3+1]))
#                 g3 = self.g(np.array((nodeXp1, nodeYp1[i3+1])))
#                 phis[3] *= g3
#                 self.gradphis[3,:,0] *= g3
#                 if i3 >= nYnodes:
#                     i3 = -2*nYnodes - nBottomNodes - 1 - np.argwhere(
#                         abs(self.DirichletNodeX[1] - nodeXp1) < 1e-10)[0,0]
#                 elif iPlane == nXnodes:
#                     i3 = -i3 - nYnodes - 1
#                 # if not np.allclose(p3, self.sim.nodes[i3]):
#                 #     print(f'p3 = ({p3[0]}, {p3[1]}),\t node = {self.sim.nodes[i3]})')
#             else:
#                 i3 += iPlane*nYnodes
#         rho = (p[0] - zetaMinus) / (zetaPlus - zetaMinus)

#         self.gradphis[:,:,1] *= phis.reshape(4,1)
#         self.gradphis[:,:,0] *= np.array((1-rho, 1-rho, rho, rho)).reshape(4,1)

#         # At this point phis = phi_FEM, so we then multiply by ramp
#         phis[0:2] *= (1 - rho)
#         phis[2:4] *= rho
#         p.shape = originalShape

#         # # these checks only true for boundary nodes if g() is unity function
#         # if (abs(np.sum(phis) - 1.0) > 1e-10):
#         #     print('Error: phis not forming partition of unity')
#         # if (abs(np.sum(self.gradphis)) > 1e-10):
#         #     print('Error: gradphis not summing to zero')
#         # print(f'p = {p}, inds = {np.array((i0, i1, i2, i3))}\n{self.nodes[np.array((i0, i1, i2, i3))]}')

#         return phis, self.gradphis.sum(axis=-1), np.array((i0, i1, i2, i3))


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
#         self.nNodes = self.nDoFs + self.nDirichletNodes

#     def computeNodes(self):
#         nDoFs = self.nDoFs
#         nYnodes = self.nYnodes
#         nodeY = self.sim.nodeY
#         self.nodes = np.empty((self.nNodes, 2))
#         self.nodes[:nDoFs] = np.vstack((
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

#     def __call__(self, p, iPlane=None):
#         if iPlane is None:
#             iPlane = np.searchsorted(self.sim.nodeX[1:], p[0])
#         originalShape = p.shape
#         p.shape = (2,)
#         nDoFs = self.nDoFs
#         NY = self.sim.NY
#         nodeX = self.sim.nodeX[iPlane]
#         nodeXp1 = self.sim.nodeX[iPlane + 1]
#         nodeY = self.sim.nodeY[iPlane]
#         nodeYp1 = self.sim.nodeY[iPlane + 1]
#         idy = self.sim.idy[iPlane]
#         idyp1 = self.sim.idy[iPlane + 1]

#         phis = np.empty(4)
#         inds = np.empty(4, dtype='int')
#         gradphis = np.empty((4,2))

#         ##### left #####
#         mapL = float(self.mapping(p, nodeX))
#         indL = (np.searchsorted(nodeY, mapL, side='right') - 1) % NY
#         phis[1] = (mapL - nodeY[indL]) * idy[indL]
#         phis[0] = 1 - phis[1]
#         gradphis[1,1] = idy[indL]
#         gradphis[0,1] = -gradphis[1,1]
#         ##### if indL on the left boundary #####
#         if (iPlane == 0):
#             g0 = self.g(np.array((nodeX, nodeY[indL])))
#             g1 = self.g(np.array((nodeX, nodeY[indL+1])))
#             phis[0] *= g0
#             phis[1] *= g1
#             gradphis[0,1] *= g0
#             gradphis[1,1] *= g1
#             inds[0] = -indL % -NY - 1
#             inds[1] = inds[0] % -NY - 1
#             # if not np.allclose(np.array((nodeX, nodeY[indL])), self.sim.nodes[inds[0]]):
#             #     print(f'p0 = ({nodeX}, {nodeY[indL]}),\t node = {self.sim.nodes[inds[0]]})')
#             # if not np.allclose(np.array((nodeX, nodeY[indL+1]%1)), self.sim.nodes[inds[1]]):
#             #     print(f'p1 = ({nodeX}, {nodeY[indL+1]%1}),\t node = {self.sim.nodes[inds[1]]})')
#         else: # indL is interior
#             inds[:2] = (indL + (iPlane-1)*NY, (indL + 1) % NY + (iPlane-1)*NY)
#             # if np.any(inds[:2] >= nDoFs):
#             #     print(f'invalid index in {inds[:2]}')

#         ##### right #####
#         mapR = float(self.mapping(p, nodeXp1))
#         indR = (np.searchsorted(nodeYp1, mapR, side='right') - 1) % NY
#         phis[3] = (mapR - nodeYp1[indR]) * idyp1[indR]
#         phis[2] = 1 - phis[3]
#         gradphis[3,1] = idyp1[indR]
#         gradphis[2,1] = -gradphis[3,1]
#         ##### if i2 on the right boundary #####
#         if (iPlane == self.nXnodes):
#             g2 = self.g(np.array((nodeXp1, nodeYp1[indR])))
#             g3 = self.g(np.array((nodeXp1, nodeYp1[indR+1])))
#             phis[2] *= g2
#             phis[3] *= g3
#             gradphis[2,1] *= g2
#             gradphis[3,1] *= g3
#             inds[2] = -indR % -NY - NY - 1
#             inds[3] = inds[2] % -NY - NY - 1
#             # if not np.allclose(np.array((nodeXp1, nodeYp1[indR])), self.sim.nodes[inds[2]]):
#             #     print(f'p2 = ({nodeXp1}, {nodeYp1[indR]}),\t node = {self.sim.nodes[inds[2]]})')
#             # if not np.allclose(np.array((nodeXp1, nodeYp1[indR+1]%1)), self.sim.nodes[inds[3]]):
#             #     print(f'p3 = ({nodeXp1}, {nodeYp1[indR+1]%1}),\t node = {self.sim.nodes[inds[3]]})')
#         else:
#             inds[2:] = ( (indR + NY*iPlane) % nDoFs,
#                          ((indR+1) % NY + NY*iPlane) % nDoFs )
#             # if np.any(inds[2:] >= nDoFs):
#             #     print(f'invalid index in {inds[2:]}')

#         gradRho = 1.0 / self.sim.dx[iPlane]
#         gradphis[:,0] = np.array((-gradRho, -gradRho, gradRho, gradRho))
#         rho = (p[0] - nodeX) * gradRho
#         rhos = np.array((1-rho, 1-rho, rho, rho))
#         gradphis[:,0] *= phis
#         gradphis[:,1] *= rhos
#         gradphis[:,0] -= self.sim.mapping.deriv(p)*gradphis[:,1]
#         # At this point phis = phi_FEM, so we multiply by ramp
#         phis *= rhos
#         p.shape = originalShape
#         return phis, gradphis, inds
