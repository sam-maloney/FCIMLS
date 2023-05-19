# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:47:07 2020

@author: Samuel A. Maloney

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sp_la

import fcimls

class sinXsinY:
    xmax = 1.
    ymax = 1.
    xfac = 2*np.pi/xmax
    yfac = 2*np.pi/ymax

    def __call__(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        return np.sin(self.xfac*x)*np.sin(self.yfac*y)

class QuadraticTestProblem:
    xmax = 1.
    ymax = 1.
    n = 3
    N = (2*np.pi/ymax)*n
    b = 0.05
    # define a such that (0, 0) maps to (xmax, 1) for given b and xmax
    a = (1 - b*xmax)/xmax**2

    def __call__(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        return x*np.sin(2*np.pi*self.n*(y - self.a*x**2 - self.b*x))

class linearPatch:
    xmax = 1.
    ymax = 1.
    umax = 1.

    def __call__(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        return 1*x + 2*y


class quadraticPatch:
    xmax = 1.
    ymax = 1.
    umax = 1.

    def __call__(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        return 0.5 + 0.1*x + 0.8*y + 1.2*x*y + 0.8*x*x + 0.6*y*y

# f = sinXsinY()
f = QuadraticTestProblem()
# f = linearPatch()
# f = quadraticPatch()

# mapping = fcimls.mappings.SinusoidalMapping(0.2, -0.25*f.xmax, f.xmax)
mapping = fcimls.mappings.QuadraticMapping(f.a, f.b)
# mapping = fcimls.mappings.LinearMapping(1/f.xmax)
# mapping = fcimls.mappings.StraightMapping()

Nratio = 1
perturbation = 0.
kwargs={
    'mapping' : mapping,
    # 'boundary' : ('Dirichlet', (1.5, f, None)),
    # # 'boundary' : ('periodic', 1.5),
    # 'basis' : 'linear',
    'boundary' : ('Dirichlet', (3.5, f, Nratio*2)),
    # 'boundary' : ('periodic', 2.5),
    'basis' : 'quadratic',
    'kernel' : 'cubic',
    'velocity' : np.array([0., 0.]),
    'diffusivity' : 1., # Makes diffusivity matrix K into Poisson operator
    'px' : perturbation,
    'py' : perturbation,
    'seed' : 42,
    'xmax' : f.xmax,
    'ymax' : f.ymax }

# allocate arrays for convergence testing
start = 5
stop = 5
nSamples = stop - start + 1
NX_array = np.logspace(start, stop, num=nSamples, base=2, dtype='int32')
E_inf = np.empty(nSamples, dtype='float64')
E_2 = np.empty(nSamples, dtype='float64')

# loop over N to test convergence where N is the number of
# grid cells along one dimension, each cell forms 2 triangles
# therefore number of nodes equals (N+1)*(N+1)
for iN, NX in enumerate(NX_array):

    # NY = NX
    NY = Nratio*NX
    
    # NQX = Nratio*3
    NQX = 1
    NQY = NY
    Qord = 3

    # allocate arrays and compute grid
    sim = fcimls.FciMlsSim(NX, NY, **kwargs)
    sim.setInitialConditions(f)
    
    print(f'NX = {NX},\tNY = {NY},\tnNodes = {sim.nNodes}')


    # Assemble the mass matrix and forcing term
    # sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationConservativeVCI
    sim.computeSpatialDiscretization(f, NQX=NQX, NQY=NQY, Qord=Qord, quadType='u',
                                     massLumping=False, vci=0)
    
    # enforce boundaries with Lagrange multipliers
    M, b = sim.boundary.modifyOperatorMatrix(sim.M, sim.b)
    uI = sp_la.spsolve(M, b)
    sim.uI = uI[:sim.nNodes]
    
    # # Sets all nodes on boundaries via strong-form co-location
    # for n, node in enumerate(sim.nodes):
    #     if (node.prod() == 0) or (node[0] == f.xmax) or (node[1] == f.ymax):
    #         inds, phis = sim.phi(node)
    #         # sim.M.data[sim.M.indptr[n]:sim.M.indptr[n+1]] = 0.
    #         sim.M[n,inds] = phis
    #         sim.b[n] = f(node)
    # sim.uI = sp_la.spsolve(sim.M, sim.b)
    
    # ##### strong form colocation at all nodes #####
    # sim.M.data[:] = 0
    # for n, node in enumerate(sim.nodes):
    #     inds, phis = sim.phi(node)
    #     # sim.M.data[sim.M.indptr[n]:sim.M.indptr[n+1]] = 0.
    #     sim.M[n,inds] = phis
    #     sim.b[n] = f(node)
    # sim.uI = sp_la.spsolve(sim.M, sim.b)
    
    sim.solve()

    # compute the analytic solution and error norms
    u_exact = f(sim.nodes)
    u_diff = u_exact - sim.u
    E_inf[iN] = np.linalg.norm(u_diff, np.inf)
    E_2[iN] = np.linalg.norm(u_diff)/np.sqrt(sim.nNodes)

    # print(f'max error = {E_inf[iN]}')
    # print(f'L2 error  = {E_2[iN]}\n')
    with np.printoptions(formatter={'float': lambda x: format(x, '.8e')}):
        print('E_2   =', repr(E_2))
        print('E_inf =', repr(E_inf), '\n', flush=True)
    
# print summary
print(f'xmax = {f.xmax}, {mapping}')
print(f'px = {kwargs["px"]}, py = {kwargs["py"]}, seed = {kwargs["seed"]}')
print(f'basis = {sim.basis.name}, kernel = {sim.kernel.name}')
print(f'boundary = {sim.boundary}')
print(f'NQX = {NQX}, NQY = {NQY//NY}*NY, Qord = {Qord}')
print(f'massLumping = {sim.massLumping}, quadType = {sim.quadType}')
print(f'VCI: {sim.vci} using {sim.vci_solver}\n')
with np.printoptions(formatter={'float': lambda x: format(x, '.8e')}):
    print('E_2     =', repr(E_2))
    print('E_inf   =', repr(E_inf))


# %% Plotting

# clear the current figure, if opened, and set parameters
fig = plt.figure(figsize=(7.75, 3))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

# SMALL_SIZE = 7
# MEDIUM_SIZE = 8
# BIGGER_SIZE = 10
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

sim.generatePlottingPoints(nx=4, ny=4)
sim.computePlottingSolution()

exact_sol = f(np.vstack((sim.X,sim.Y)).T)
error = exact_sol - sim.U
maxAbsErr = np.max(np.abs(error))
vmin = -maxAbsErr
vmax = maxAbsErr

ax1 = plt.subplot(121)
field = ax1.tripcolor(sim.X, sim.Y, error, shading='gouraud'
                      ,cmap='RdBu', vmin=vmin, vmax=vmax
                      )
x = np.linspace(0, sim.nodeX[-1], 100)
if mapping.name == 'quadratic':
    startingPoints = [0.]
else:
    startingPoints = [0.4, 0.5, 0.6]
for yi in startingPoints:
    ax1.plot(x, [sim.boundary.mapping(np.array([[0, yi]]), i) for i in x], 'k')
# for xi in sim.nodeX:
#     ax1.plot([xi, xi], [0, 1], 'k:')
# ax.plot(sim.X[np.argmax(sim.U)], sim.Y[np.argmax(sim.U)],
#   'g+', markersize=10)
# cbar = plt.colorbar(field, format='%.0e')
cbar = plt.colorbar(field)
cbar.formatter.set_powerlimits((0, 0))
plt.xlabel(r'$x$')
plt.ylabel(r'$y$', rotation=0)
if abs(f.xmax - 2*np.pi) < 1e-10:
    plt.xticks(np.linspace(0, f.xmax, 5),
        ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
#  plt.xticks(np.linspace(0, 2*np.pi, 7),
#      ['0',r'$\pi/3$',r'$2\pi/3$',r'$\pi$',r'$4\pi/3$',r'$5\pi/3$',r'$2\pi$'])
else:
    plt.xticks(np.linspace(0, f.xmax, 6))
plt.margins(0,0)

# plot the error convergence
ax1 = plt.subplot(122)
plt.loglog(NX_array, E_inf, '.-', label=r'$E_\infty$ magnitude')
plt.loglog(NX_array, E_2, '.-', label=r'$E_2$ magnitude')
plt.minorticks_off()
plt.xticks(NX_array, NX_array)
plt.xlabel(r'$NX$')
plt.ylabel(r'Magnitude of Error Norm')

# plot the intra-step order of convergence
ax2 = ax1.twinx()
logN = np.log(NX_array)
logE_inf = np.log(E_inf)
logE_2 = np.log(E_2)
order_inf = (logE_inf[0:-1] - logE_inf[1:])/(logN[1:] - logN[0:-1])
order_2 = (logE_2[0:-1] - logE_2[1:])/(logN[1:] - logN[0:-1])
intraN = np.logspace(start+0.5, stop-0.5, num=nSamples-1, base=2.0)
plt.plot(intraN, order_inf, '.:', linewidth=1, label=r'$E_\infty$ order')
plt.plot(intraN, order_2, '.:', linewidth=1, label=r'$E_2$ order')
plt.plot(plt.xlim(), [2, 2], 'k:', linewidth=1, label='Expected')
plt.ylim(0, 5)
plt.yticks(np.linspace(0,5,6))
plt.ylabel(r'Intra-step Order of Convergence')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='best')
plt.margins(0,0)

# fig.savefig("CD_MassLumped_RK4.pdf", bbox_inches = 'tight', pad_inches = 0)
