# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:47:07 2020

@author: Samuel A. Maloney

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la

import fcimls

from timeit import default_timer

class QuadraticTestProblem:
    xmax = 1.
    ymax = 1.
    umax = xmax
    n = 3
    N = (2*np.pi/ymax)*n
    # a = 0.01
    b = 0.05
    # define a such that (0, 0) maps to (xmax, 1) for given b and xmax
    a = (1 - b*xmax)/xmax**2

    def __call__(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        N = self.N
        a = self.a
        b = self.b
        return N*(N*x*(4*a**2*x**2 + 4*a*b*x + b**2 + 1)*np.sin(N*(y - a*x**2 - b*x))
                  + 2*(3*a*x + b)*np.cos(N*(y - a*x**2 - b*x)))

    def solution(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        return x*np.sin(self.N*(y - self.a*x**2 - self.b*x))


class slantedTestProblem:
    xmax = 1.
    ymax = 1.
    n = 8

    xfac = 2*np.pi/xmax
    yfac = 2*np.pi/ymax
    n2 = n*n
    yf2 = yfac*yfac
    _2nyf2 = 2*n*yf2
    n2xf2pyf2 = n2*(xfac*xfac + yf2)
    n2xf2pyf2pyf2 = n2xf2pyf2 + yf2

    A = 0.5 / n2xf2pyf2
    B = 0.5 / (n2xf2pyf2pyf2 - _2nyf2*_2nyf2/n2xf2pyf2pyf2)
    C = B*_2nyf2 / n2xf2pyf2pyf2

    aA = abs(A)
    aB = abs(B)
    aC = abs(C)
    umax = aA + aB + aC

    def __call__(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        yarg = self.yfac*y
        return 0.5*np.sin(self.n*(yarg - self.xfac*x))*(1 + np.sin(yarg))

    def solution(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        yarg = self.yfac*y
        xyarg = self.n*(yarg - self.xfac*x)
        return self.A*np.sin(xyarg) + self.B*np.sin(yarg)*np.sin(xyarg) \
                                    + self.C*np.cos(yarg)*np.cos(xyarg)


class simplifiedSlantProblem:
    xmax = 1.
    ymax = 1.
    n = 2

    xfac = 2*np.pi/xmax
    yfac = 2*np.pi/ymax
    umax = 1/(2*n*n*(yfac*yfac + xfac*xfac))

    def __call__(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        return 0.5*np.sin(self.n*(self.yfac*y - self.xfac*x))

    def solution(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        return self.umax * np.sin(self.n*(self.yfac*y - self.xfac*x))


class sinXsinY:
    xmax = 1.
    ymax = 1.
    xfac = 2*np.pi/xmax
    yfac = 2*np.pi/ymax
    umax = (1 / (xfac**2 + yfac**2))

    def __call__(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        return np.sin(self.xfac*x)*np.sin(self.yfac*y)

    def solution(self, p):
        return self.umax * self(p)


class linearPatch:
    xmax = 1.
    ymax = 1.
    umax = xmax + 2*ymax
    
    b = 0.05
    # define a such that (0, 0) maps to (xmax, 1) for given b and xmax
    a = (1 - b*xmax)/xmax**2

    def __call__(self, p):
        nPoints = p.size // 2
        return np.zeros(nPoints)

    def solution(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        return 1*x + 2*y


class quadraticPatch:
    xmax = 1.
    ymax = 1.

    xx = 0.8
    yy = 0.6
    
    umax = 0.5 + 0.1*xmax + 0.8*ymax + 1.2*xmax*ymax + xx*xmax*xmax + yy*ymax*ymax
    
    b = 0.05
    # define a such that (0, 0) maps to (xmax, 1) for given b and xmax
    a = (1 - b*xmax)/xmax**2

    def __call__(self, p):
        nPoints = p.size // 2
        return np.full(nPoints, -2*(self.xx + self.yy))

    def solution(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        return 0.5 + 0.1*x + 0.8*y + 1.2*x*y + self.xx*x*x + self.yy*y*y

# f = QuadraticTestProblem()
# f = slantedTestProblem()
# f = simplifiedSlantProblem()
f = sinXsinY()
# f = linearPatch()
# f = quadraticPatch()

mapping = fcimls.mappings.SinusoidalMapping(0.2, -0.25*f.xmax, f.xmax)
# mapping = fcimls.mappings.QuadraticMapping(f.a, f.b)
# mapping = fcimls.mappings.LinearMapping(1/f.xmax)
# mapping = fcimls.mappings.StraightMapping()


Nratio = 2
perturbation = 0.1
kwargs={
    'mapping' : mapping,
    # 'boundary' : ('Dirichlet', (1.5, f.solution, None)),
    # 'boundary' : ('periodic', 1.5),
    # 'basis' : 'linear',
    # 'boundary' : ('Dirichlet', (4.5, f.solution, 2*Nratio)),
    'boundary' : ('periodic', 2.5),
    'basis' : 'quadratic',
    # 'kernel' : 'cubic',
    # 'kernel' : 'quartic',
    'kernel' : 'quintic',
    # 'kernel' : 'septic',
    # 'kernel' : fcimls.kernels.GenericSpline(n=5),
    # 'kernel' : 'bump',
    'velocity' : np.array([0., 0.]),
    'diffusivity' : 1., # Makes diffusivity matrix K into Poisson operator
    'px' : perturbation,
    'py' : perturbation,
    'seed' : 42,
    'xmax' : f.xmax,
    'ymax' : f.ymax }

# allocate arrays for convergence testing
start = 2
stop = 3
nSamples = np.rint(stop - start + 1).astype('int')
NX_array = np.logspace(start, stop, num=nSamples, base=2, dtype='int')
E_inf = np.empty(nSamples)
E_2 = np.empty(nSamples)
t_setup = np.empty(nSamples)
t_solve = np.empty(nSamples)
dxi = []

print('Poisson_test.py\n')

# loop over N to test convergence where N is the number of
# grid cells along one dimension, each cell forms 2 triangles
# therefore number of nodes equals (N+1)*(N+1)
for iN, NX in enumerate(NX_array):

    start_time = default_timer()

    # Nratio = 16

    NY = NX*Nratio
    # NX = 16

    # NQX = Nratio // 2
    NQX = 1
    NQY = NY
    Qord = 4

    ##### allocate arrays and compute grid
    sim = fcimls.FciMlsSim(NX, NY, **kwargs)
    
    # ##### Add extra node(s)
    # extraNodes = np.array([[1 - 0.5*sim.dx[-1], 0.5*sim.dy[-1,0]]])
    # sim.boundary.nNodes += len(extraNodes)
    # sim.nNodes = sim.boundary.nNodes
    # sim.boundary.nodes = np.vstack((extraNodes, sim.nodes))
    # sim.nodes = sim.boundary.nodes
    # sim.nodesMapped = sim.nodes.copy()
    # sim.nodesMapped[:,1] = sim.boundary.mapping(sim.nodes, 0)
    # sim.boundary.isBoundaryNode = np.concatenate(
    #     (np.full(len(extraNodes), False), sim.boundary.isBoundaryNode) )

    print(f'NX = {NX},\tNY = {NY},\tnNodes = {sim.nNodes}')

    ##### Assemble the mass matrix and forcing term
    # sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationConservativeVCI6
    # sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationConservativeVCI
    sim.computeSpatialDiscretization(f, NQX=NQX, NQY=NQY, Qord=Qord, quadType='u',
                                     massLumping=False, vci=2)
    K, b = sim.boundary.modifyOperatorMatrix(sim.K, sim.b)

    # if sim.boundary.name == 'Dirichlet':
    #     ##### Enforce exact solution constraints directly #####
    #     # Sets all nodes on boundaries via strong-form co-location
    #     for n, node in enumerate(sim.nodes):
    #         if (node.prod() == 0) or (node[0] == f.xmax) or (node[1] == f.ymax):
    #             inds, phis = sim.phi(node)
    #             sim.K.data[sim.K.indptr[n]:sim.K.indptr[n+1]] = 0.
    #             sim.K[n,inds] = phis
    #             sim.b[n] = f.solution(sim.nodes[n])
    # K = sim.K
    # b = sim.b

    if sim.boundary.name == 'periodic':
        ##### Enforce exact solution constraints directly #####
        # ### Sets all nodes on x and y axes via strong-form co-location
        # for n, node in enumerate(sim.nodes):
        #     if node.prod() == 0.:
        #         inds, phis = sim.phi(node)
        #         sim.K.data[sim.K.indptr[n]:sim.K.indptr[n+1]] = 0.
        #         sim.K[n,inds] = phis
        #         sim.b[n] = f.solution(node)
        
        ### Sets all nodes on x and y axes using Lagrange multipliers
        isAxesNode = sim.nodes.prod(axis=1) == 0
        nAxesNodes = np.count_nonzero(isAxesNode)
        nMaxEntries = int((sim.nNodes * sim.boundary.volume)**2 * nAxesNodes)
        data = np.zeros(nMaxEntries)
        row_ind = np.zeros(nMaxEntries, dtype='uint32')
        col_ind = np.zeros(nMaxEntries, dtype='uint32')
        index = 0
        axesNodes = sim.nodes[isAxesNode]
        for iNode, node in enumerate(axesNodes):
            indices, phis = sim.phi(node)
            nEntries = indices.size
            data[index:index+nEntries] = phis
            row_ind[index:index+nEntries] = indices
            col_ind[index:index+nEntries] = np.repeat(iNode, nEntries)
            index += nEntries
        inds = np.flatnonzero(data.round(decimals=14,out=data))
        G = sp.csr_matrix( (data[inds], (row_ind[inds], col_ind[inds])),
                            shape=(sim.nNodes, nAxesNodes) )
        axesValues = f.solution(axesNodes)
        K = sp.bmat([[sim.K, G], [G.T, None]], format='csr')
        b = np.concatenate((sim.b, axesValues))

    t_setup[iN] = default_timer()-start_time
    print(f'setup time = {t_setup[iN]:.8e} s')
    start_time = default_timer()

    ##### Solve for the approximate solution
    # tolerance = 1e-10
    # uI, info = sp_la.lgmres(K, b, tol=tolerance, atol=tolerance)
    uI = sp_la.spsolve(K, b)
    sim.uI = uI[:sim.nNodes]
    sim.solve()

    t_solve[iN] = default_timer()-start_time
    print(f'solve time = {t_solve[iN]:.8e} s')
    start_time = default_timer()

    ##### compute the analytic solution and normalized error norms
    uExact = f.solution(sim.nodes)
    E_inf[iN] = np.linalg.norm(sim.u - uExact, np.inf) / f.umax
    E_2[iN] = np.linalg.norm(sim.u - uExact)/np.sqrt(sim.nNodes) / f.umax

    # print(f'max error  = {E_inf[iN]:.8e}')
    # print(f'L2 error   = {E_2[iN]:.8e}\n', flush=True)
    with np.printoptions(formatter={'float': lambda x: format(x, '.8e')}):
        print('E_2   =', repr(E_2))
        print('E_inf =', repr(E_inf), '\n', flush=True)

# print summary
print(f'xmax = {f.xmax}, {mapping}')
print(f'px = {kwargs["px"]}, py = {kwargs["py"]}, seed = {kwargs["seed"]}')
print(f'basis = {sim.basis.name}, kernel = {sim.kernel.name}')
print(f'boundary = {sim.boundary}')
print(f'NQX = {NQX}, NQY = {NQY//NY}*NY, massLumping = {sim.massLumping}')
print(f'Qord = {Qord}, quadType = {sim.quadType}')
print(f'VCI: {sim.vci} using {sim.vci_solver}\n')
with np.printoptions(formatter={'float': lambda x: format(x, '.8e')}):
    print('E_2     =', repr(E_2))
    print('E_inf   =', repr(E_inf))
    print('t_setup =', repr(t_setup))
    print('t_solve =', repr(t_solve))
    # print(E_2[0])
    # print(E_inf[0])


# %% Plotting

plt.rc('pdf', fonttype=42)
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# fontsize : int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
# plt.rc('font', size='small')
# plt.rc('legend', fontsize='small')
# plt.rc('axes', titlesize='medium', labelsize='medium')
# plt.rc('xtick', labelsize='small')
# plt.rc('ytick', labelsize='small')
# plt.rc('figure', titlesize='large')

# clear the current figure, if opened, and set parameters
fig = plt.figure(figsize=(7.75, 3))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

# sim.generatePlottingPoints(nx=1, ny=1)
sim.generatePlottingPoints(nx=int(max(NY/NX,1)), ny=int(max(NX/NY,1)))
sim.computePlottingSolution()

# vmin = np.min(sim.U)
# vmax = np.max(sim.U)

exactSol = f.solution(np.vstack((sim.X,sim.Y)).T)
error = sim.U - exactSol
maxAbsErr = np.max(np.abs(error))
# maxAbsErr = np.max(np.abs(sim.u - uExact))
vmin = -maxAbsErr
vmax = maxAbsErr

ax1 = plt.subplot(121)
ax1.set_title('Final Solution')
field = ax1.tripcolor(sim.X, sim.Y, error, shading='gouraud'
                        ,cmap='seismic', vmin=vmin, vmax=vmax)
# field = ax1.tripcolor(sim.nodes[:,0], sim.nodes[:,1], sim.u - uExact
#                     ,shading='gouraud', cmap='seismic', vmin=vmin, vmax=vmax)
# field = ax1.tripcolor(sim.nodes[:,0], sim.nodes[:,1], sim.u, shading='gouraud')
# field = ax1.tripcolor(sim.X, sim.Y, sim.U, shading='gouraud')
# field = ax1.tripcolor(sim.X, sim.Y, exactSol, shading='gouraud')
# field = ax1.tripcolor(sim.X, sim.Y, f(np.vstack((sim.X,sim.Y)).T), shading='gouraud')
x = np.linspace(0, sim.nodeX[-1], 100)
if mapping.name == 'quadratic':
    startingPoints = [0.]
else:
    startingPoints = [0.4, 0.5, 0.6]
for yi in startingPoints:
    # try:
        ax1.plot(x, [sim.boundary.mapping(np.array([[0, yi]]), i) for i in x], 'k')
    # except:
    #     ax1.plot(x, [sim.boundary.mapping(np.array([[0, yi]]), i) % 1 for i in x], 'k')
# for xi in sim.nodeX:
#     ax1.plot([xi, xi], [0, 1], 'k:')
# ax.plot(sim.X[np.argmax(sim.U)], sim.Y[np.argmax(sim.U)],
#   'g+', markersize=10)
# cbar = plt.colorbar(field, format='%.0e')
cbar = plt.colorbar(field)
cbar.formatter.set_powerlimits((0, 0))
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$', rotation=0)
if abs(f.xmax - 2*np.pi) < 1e-10:
    ax1.set_xticks(np.linspace(0, f.xmax, 5),
        ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
#  plt.xticks(np.linspace(0, 2*np.pi, 7),
#      ['0',r'$\pi/3$',r'$2\pi/3$',r'$\pi$',r'$4\pi/3$',r'$5\pi/3$',r'$2\pi$'])
else:
    ax1.set_xticks(np.linspace(0, f.xmax, 6))
ax1.margins(0,0)

# plot the error convergence
ax2 = plt.subplot(122)
logN = np.log(NX_array)
ax2.semilogy(logN, E_inf, '.-', label=r'$E_\infty$')
ax2.semilogy(logN, E_2, '.-', label=r'$E_2$')
# ax2.minorticks_off()
ax2.set_xticks(logN, labels=NX_array)
ax2.set_xlabel(r'$NX$')
ax2.set_ylabel(r'Magnitude of Error Norm')

# plot the intra-step order of convergence
ax2R = ax2.twinx()
logE_inf = np.log(E_inf)
logE_2 = np.log(E_2)
order_inf = (logE_inf[0:-1] - logE_inf[1:])/(logN[1:] - logN[0:-1])
order_2 = (logE_2[0:-1] - logE_2[1:])/(logN[1:] - logN[0:-1])
intraN = 0.5 * (logN[:-1] + logN[1:])
ax2R.plot(intraN, order_inf, '.:', linewidth=1, label=r'$E_\infty$ order')
ax2R.plot(intraN, order_2, '.:', linewidth=1, label=r'$E_2$ order')
ax2R.axhline(2, linestyle=':', color='k', linewidth=1, label='Expected')
ax2R.set_ylim(0, 5)
ax2R.set_yticks(np.linspace(0,5,6))
# ax2R.set_lim(0, 3)
# ax2R.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
ax2R.set_ylabel(r'Intra-step Order of Convergence')
ax2.legend()
# lines, labels = ax1.get_legend_handles_labels()
