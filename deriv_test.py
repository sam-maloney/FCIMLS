#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 09:27:41 2021

@author: Samuel A. Maloney
"""

import numpy as np
import fcimls

class UnityFunction:
    xmax = 1.

    def __call__(self, p):
        return np.ones(p.size // 2)

    def solution(self, p):
        return np.ones(p.size // 2)

f = UnityFunction()

a = 0.95
b = 0.05

mapping = fcimls.mappings.SinusoidalMapping(0.2, -0.25*f.xmax, f.xmax)
# mapping = fcimls.mappings.QuadraticMapping(a, b)
# mapping = fcimls.mappings.LinearMapping(1/f.xmax)
# mapping = fcimls.mappings.StraightMapping()

perturbation = 0.
kwargs={
    'mapping' : mapping,
    # 'boundary' : ('Dirichlet', (1.5, f.solution, None)),
    # # 'boundary' : ('periodic', 1.5),
    # 'basis' : 'linear',
    'boundary' : ('Dirichlet', (2.5, f.solution, None)),
    # 'boundary' : ('periodic', 2.5),
    'basis' : 'quadratic',
    'kernel' : 'cubic',
    'velocity' : np.array([0., 0.]),
    'diffusivity' : 1., # Makes diffusivity matrix K into Poisson operator
    'px' : perturbation,
    'py' : perturbation,
    'seed' : 42,
    'xmax' : 1.,
    'ymax' : 1. }

NX = 8
NY = NX

# initialize simulation class
sim = fcimls.FciMlsSim(NX, NY, **kwargs)

nPoints = 1000
seed = 42
rtol = 1e-5
atol = 1e-6
phitol = 1e-10

dx = 1e-6
dy = dx


prng = np.random.default_rng(seed=seed)
points = prng.random((nPoints, 2))
inds = []
phis = []
gradphis = []
gradphisNum = []

for i, point in enumerate(points):
    local_inds, local_phis, local_gradphis = sim.dphi(point)
    inds.append(local_inds)
    phis.append(local_phis)
    gradphis.append(local_gradphis)
    phiError = np.abs(local_phis.sum() - 1)
    if phiError > phitol:
        print(f'No partition of unity for point {i} = {point}, error = {phiError}')
    try:
        tmp_inds, phiR = sim.phi(point + (dx, 0.))
        np.testing.assert_array_equal(inds[i], tmp_inds)
        tmp_inds, phiL = sim.phi(point - (dx, 0.))
        np.testing.assert_array_equal(inds[i], tmp_inds)
        tmp_inds, phiU = sim.phi(point + (0., dy))
        np.testing.assert_array_equal(inds[i], tmp_inds)
        tmp_inds, phiD = sim.phi(point - (0., dy))
        np.testing.assert_array_equal(inds[i], tmp_inds)
    except(AssertionError):
        gradphisNum.append(np.nan)
        print(f'index mismatch for point {i} = {point}')
        continue
    local_gradphisNum = np.empty(local_gradphis.shape)
    local_gradphisNum[:,0] = (phiR - phiL) / (2*dx)
    local_gradphisNum[:,1] = (phiU - phiD) / (2*dy)
    gradphisNum.append(local_gradphisNum)
    try:
        np.testing.assert_allclose(gradphis[i], gradphisNum[i],
                                    rtol=rtol, atol = atol)
    except(AssertionError):
        print(f'Gradient mismatch for point {i} = {point}, '
              f'max error = {np.max(abs(gradphis[i]-gradphisNum[i]))}')
