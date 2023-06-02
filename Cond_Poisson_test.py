# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:47:07 2020

@author: Samuel A. Maloney

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la

import fcimls


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


def cond(A, order=2):
    """Compute the condition number of the given matrix A.

    Parameters
    ----------
    A : {scipy.sparse.spmatrix, numpy.ndarray}
        Matrix whose condition number is computed, can be sparse or dense.
    order : {int, inf, -inf, ‘fro’}, optional
        Order of the norm. inf means numpy’s inf object. The default is 2.

    Returns
    -------
    c : float
        The condition number of the matrix.

    """
    if order == 2:
        LM = sp_la.svds(A, 1, which='LM', return_singular_vectors=False)
        SM = sp_la.svds(A, 1, which='SM', return_singular_vectors=False)
        c = LM[0]/SM[0]
    else:
        if sp.issparse(A):
            c = sp_la.norm(A, order) * sp_la.norm(sp_la.inv(A), order)
        else: # A is dense
            c = la.norm(A, order) * la.norm(la.inv(A), order)
    return c


# f = slantedTestProblem()
# f = simplifiedSlantProblem()
f = sinXsinY()

# mapping = fcimls.mappings.SinusoidalMapping(0.2, -0.25*f.xmax, f.xmax)
# mapping = fcimls.mappings.LinearMapping(1/f.xmax)
mapping = fcimls.mappings.StraightMapping()

perturbation = 0.
kwargs={
    'mapping' : mapping,
    # 'basis' : 'linear',
    'basis' : 'quadratic',
    # 'kernel' : 'cubic',
    # 'kernel' : 'quartic',
    'kernel' : 'quintic',
    # 'kernel' : 'bump',
    'velocity' : np.array([0., 0.]),
    'diffusivity' : 1., # Makes diffusivity matrix K into Poisson operator
    'px' : perturbation,
    'py' : perturbation,
    'seed' : 42,
    'xmax' : f.xmax,
    'ymax' : f.ymax }

NX = 4
Nratio = 1
NY = NX*Nratio

NQX = 1
NQY = NY
Qord = 4

# allocate arrays for condition number testing
start = 1.5
stop = 9.5
step = 0.1
nSamples = int(np.rint((stop - start)/step)) + 1
support_size_array = np.linspace(start, stop, num=nSamples)
E_inf = np.empty(nSamples)
E_2 = np.empty(nSamples)
# cond_without_BCs = np.empty(nSamples)
cond_with_BCs = np.empty(nSamples)
dxi = []

print('Cond_Poisson_test.py\n')
print(f'NX = {NX},\tNY = {NY},\tnNodes = {NX*NY}\n')

# loop over N to test convergence where N is the number of
# grid cells along one dimension, each cell forms 2 triangles
# therefore number of nodes equals (N+1)*(N+1)
for iD, size in enumerate(support_size_array):

    ##### allocate arrays and compute grid
    sim = fcimls.FciMlsSim(NX, NY, boundary=('periodic', size), **kwargs)
    

    print(f'size = {size}')

    ##### Assemble the mass matrix and forcing term
    # sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationConservativeVCI6
    sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationConservativeVCI
    sim.computeSpatialDiscretization(f, NQX=NQX, NQY=NQY, Qord=Qord, quadType='u',
                                     massLumping=False, vci=2)

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
    
    
    # cond_without_BCs[iD] = cond(sim.K)
    cond_with_BCs[iD] = cond(K)


    ##### Solve for the approximate solution
    # tolerance = 1e-10
    # uI, info = sp_la.lgmres(K, b, tol=tolerance, atol=tolerance)
    uI = sp_la.spsolve(K, b)
    sim.uI = uI[:sim.nNodes]
    sim.solve()


    ##### compute the analytic solution and normalized error norms
    uExact = f.solution(sim.nodes)
    E_inf[iD] = np.linalg.norm(sim.u - uExact, np.inf) / f.umax
    E_2[iD] = np.linalg.norm(sim.u - uExact)/np.sqrt(sim.nNodes) / f.umax

    # print(f'c_without  = {cond_without_BCs[iD]:.8e}')
    print(f'c_with     = {cond_with_BCs[iD]:.8e}')
    print(f'max error  = {E_inf[iD]:.8e}')
    print(f'L2 error   = {E_2[iD]:.8e}\n', flush=True)
    # with np.printoptions(formatter={'float': lambda x: format(x, '.8e')}):
    #     print('c_w/o  =', repr(cond_without_BCs))
    #     print('c_with =', repr(cond_with_BCs))
    #     print('E_2    =', repr(E_2))
    #     print('E_inf  =', repr(E_inf), '\n', flush=True)

# print summary
print(f'xmax = {f.xmax}, {mapping}')
print(f'px = {kwargs["px"]}, py = {kwargs["py"]}, seed = {kwargs["seed"]}')
print(f'basis = {sim.basis.name}, kernel = {sim.kernel.name}')
print(f'boundary = {sim.boundary}')
print(f'NQX = {NQX}, NQY = {NQY//NY}*NY, massLumping = {sim.massLumping}')
print(f'Qord = {Qord}, quadType = {sim.quadType}')
print(f'VCI: {sim.vci} using {sim.vci_solver}\n')
with np.printoptions(formatter={'float': lambda x: format(x, '.8e')}):
    # print('c_w/o  =', repr(cond_without_BCs))
    print('c_with =', repr(cond_with_BCs))
    print('E_2     =', repr(E_2))
    print('E_inf   =', repr(E_inf))
    # print(E_2[0])
    # print(E_inf[0])


# # %% Plotting

# plt.rc('pdf', fonttype=42)
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage[T1]{fontenc}'
#                                 r'\usepackage[osf,largesc]{newpxtext}'
#                                 # r'\usepackage[osf,nohelv,largesc]{newpxtext}'
#                                 r'\usepackage[euler-digits]{eulervm}'
#                                 # r'\usepackage{eulerpx}'
#                                 # r'\usepackage[sans]{libertinus}'
#                                 r'\usepackage{classico}'
#                                 r'\usepackage{mathtools}'
#                                 r'\newcommand*{\norm}[1]{\left\lVert#1\right\rVert}'
#                                 )
# plt.rc('font', family='sans-serif')
# # plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# # fontsize : int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
# # plt.rc('font', size='small')
# plt.rc('legend', fontsize='small')
# # plt.rc('axes', titlesize='medium', labelsize='medium')
# # plt.rc('xtick', labelsize='small')
# # plt.rc('ytick', labelsize='small')
# # plt.rc('figure', titlesize='large')

# # clear the current figure, if opened, and set parameters
# fig = plt.figure(figsize=(7.75, 3))
# axes = fig.subplots(1,2)
# fig.subplots_adjust(hspace=0.3, wspace=0.3)

# axes[0].plot(support_size_array, cond_with_BCs, 'ko-')
# axes[0].set_ylabel('condition number')
# axes[0].set_ylim((0,2000))

# axes[1].plot(support_size_array, E_2, 'ko-')
# axes[1].set_ylabel(r'$\norm{u-u^d}$')
# axes[1].set_ylim((0,2))

# for ax in axes:
#     # for size in [3,4,5]:
#     #     ax.axvline(size, linestyle=':', color='black')#, linewidth=dashed_linewidth)
    
#     # ax.set_xlabel('support size')
#     ax.set_xticks(np.arange(np.trunc(start), np.ceil(stop)+1).astype('int'))

# fig.suptitle(f'{mapping}, p = {perturbation}')
# fig.supxlabel('support size as a multiple of uniform grid spacing', verticalalignment='top')
