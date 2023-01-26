#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:25:47 2020

@author: Samuel A. Maloney
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import fcimls

def gaussian(points):
    A = 1.0
    ndim = points.shape[1]
    r0 = (0.5, 0.5, 0.5)[0:ndim]
    sigma = (0.1, 0.1, 0.1)[0:ndim]
    return np.exp( -0.5*A*np.sum(((points - r0)/sigma )**2, 1) )

def hat(points):
    return np.hstack((points > 0.25, points < 0.75)).all(1).astype('float64')

n = 100
xmax = 1.

mapping = fcimls.mappings.SinusoidalMapping(0.2, -0.25*xmax, xmax)
# mapping = fcimls.mappings.LinearMapping(1/xmax)
# mapping = fcimls.mappings.StraightMapping()

perturbation = 0.
kwargs={
    'mapping' : mapping,
    'boundary' : ('Dirichlet', (1.5, None, None)),
    # 'boundary' : ('periodic', 1.5),
    'basis' : 'linear',
    # 'boundary' : ('periodic', 2.5),
    # 'basis' : 'quadratic',
    'kernel' : 'cubic',
    # 'kernel' : 'quartic',
    # 'kernel' : 'Gaussian',
    'velocity' : np.array([0., 0.]),
    'diffusivity' : 1., # Makes diffusivity matrix K into Poisson operator
    'px' : perturbation,
    'py' : perturbation,
    'seed' : 42,
    'xmax' : 1.,
    'ymax' : 1. }

precon='ilu'
tolerance = 1e-10

# Initialize simulation
NX = 4
NY = 4
sim = fcimls.FciMlsSim(NX=NX, NY=NY, **kwargs)

points = ( np.indices(np.repeat(n+1, 2), dtype='float64')
           .T.reshape(-1,2) ) / n
phis = np.zeros((len(points), sim.nNodes))
gradphis = np.zeros((len(points), sim.nNodes, 2))
for i, point in enumerate(points):
    inds, local_phis, local_gradphis = sim.dphi(point)
    phis[i, inds] = local_phis
    gradphis[i, inds] = local_gradphis

# open new figure
fig = plt.figure(figsize=(15, 13))
mpl.rc('axes', titlesize='xx-large', labelsize='x-large')
mpl.rc('xtick', labelsize='large')
mpl.rc('ytick', labelsize='large')

plt.subplots_adjust(hspace = 0.3, wspace = 0.2)

plotVar = phis
# plotVar = gradphis[:,:,0]
# plotVar = gradphis[:,:,1]
maxAbs = np.max(np.abs(plotVar))

if sim.boundary.name == 'periodic':
    nx = NX
    ny = NY
elif sim.boundary.name == 'Dirichlet':
    nx = NX + 1
    ny = NY + 1

for j in range(ny):
    for i in range(nx):
        # plot the result
        plt.subplot(ny,nx,nx*ny-(j+1)*nx+i+1)
        plt.tripcolor(points[:,0], points[:,1], plotVar[:,i*ny+j],
            shading='gouraud', vmin=-maxAbs , vmax=maxAbs, cmap='seismic')
        plt.colorbar()
        if j == 0:
            plt.xlabel(r'$x$')
        if i == 0:
            plt.ylabel(r'$y$', rotation=0)
        plt.title('$\Phi_{{{0}}}$'.format(i*ny+j))
        plt.xticks([0, 1])
        plt.yticks([0, 1])
        # plt.xticks([0.0, 0.5, 1.0])
        # plt.yticks([0.0, 0.5, 1.0])
        # plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.margins(0,0)

# plt.savefig('MLS_shape_functions5.pdf', bbox_inches='tight', pad_inches=0)