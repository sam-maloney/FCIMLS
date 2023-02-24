# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:46:29 2021

@author: Samuel A. Maloney
"""

import numpy as np
import matplotlib.pyplot as plt

E_2 = []
E_inf = []
t_setup = []
t_solve = []
labels = []
NX = []
NY = []

##### VCI-C (whole domain) #####

# For all of the below, unless otherwise noted
# points on axes constrained via Lagrange multipliers
# f(x,y) = 0.5*sin(n*(2pi*y - 2pi*x))*(1 + sin(2pi*y))
# n = 8
# Omega = (0,1) X (0,1)

# px = 0.1, py = 0.1, seed = 42
# basis = quadratic, kernel = cubic
# boundary = PeriodicBoundary(support = [2.5 2.5])
# NQY = 1*NY
# massLumping = False


##### NQX = 1, Qord = 3, quadType = Gauss-Legendre

# # StraightMapping()
# # VCI: None using None
# E_2.append(np.array([8.82567144e+00, 2.77912929e-01, 1.55893826e-01, 1.02159737e-02,
#        3.11636630e-03, 9.88205497e-04, 4.08445565e-04]))
# E_inf.append(np.array([1.84377958e+01, 7.22907319e-01, 4.52238763e-01, 4.74116020e-02,
#        1.58404367e-02, 5.40643611e-03, 2.89174725e-03]))
# labels.append('unaligned 1:1, no VCI')
# NX.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY.append(1)

# # StraightMapping()
# # VCI: VC2-C (whole domain) using ssqr.min2norm
# E_2.append(np.array([2.17477562e+01, 6.20306535e+00, 4.95240407e-01, 3.18621101e-02,
#        3.02674879e-03, 4.07198399e-04]))
# E_inf.append(np.array([5.45452786e+01, 1.95737399e+01, 1.73450007e+00, 1.71463078e-01,
#        1.68801072e-02, 2.85822630e-03]))
# labels.append('unaligned 1:1, VC2-C')
# NX.append(np.array([  4,   8,  16,  32,  64, 128]))
# NY.append(1)


##### VCI: VC2-C (whole domain) using ssqr.min2norm

# # StraightMapping(), NQX = 2, Qord = 2, quadType = Gauss-Legendre
# E_2.append(np.array([4.55303179e+01, 4.87389007e+01, 1.97373530e+00, 5.99196411e-01,
#        3.09522991e-01, 3.68915079e-02, 2.35175102e-03]))
# E_inf.append(np.array([1.19104535e+02, 1.13987341e+02, 7.06690978e+00, 4.61815112e+00,
#        7.91826940e+00, 9.85003330e-01, 1.39306965e-01]))
# labels.append('unaligned 1:1, NQX=2,2g')
# NX.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY.append(1)

# # LinearMapping(1.0), NQX = 2, Qord = 2, quadType = Gauss-Legendre
# E_2.append(np.array([5.06105761e+01, 1.57136186e+01, 2.45170387e-01, 6.91172954e-03,
#        1.60971161e-03, 2.88747184e-04, 5.52500522e-05]))
# E_inf.append(np.array([1.14438650e+02, 4.49788670e+01, 8.94738349e-01, 4.03158752e-02,
#        9.09576612e-03, 1.26080516e-03, 2.05104334e-04]))
# labels.append('aligned 1:1, NQX=2,2g')
# NX.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY.append(1)

# StraightMapping(), NQX = 2, Qord = 3, quadType = uniform
E_2.append(np.array([3.66500704e+00, 1.79102106e+00, 2.44918893e-01, 1.60849314e-02,
       2.47328472e-03, 4.55351099e-04, 8.58986774e-05]))
E_inf.append(np.array([1.15697486e+01, 6.40099115e+00, 8.48972481e-01, 7.51296999e-02,
       1.13340149e-02, 2.39105291e-03, 3.65807720e-04]))
labels.append('unaligned 1:1, NQX=2,3u')
NX.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY.append(1)

# LinearMapping(1.0), NQX = 2, Qord = 3, quadType = uniform
E_2.append(np.array([4.50816969e+00, 4.35767178e+00, 2.27755332e-01, 5.66164282e-03,
       1.30324143e-03, 1.92936997e-04, 2.43886935e-05]))
E_inf.append(np.array([1.12285705e+01, 1.16367501e+01, 8.02927946e-01, 2.25277766e-02,
       8.54735061e-03, 1.13792309e-03, 1.56774456e-04]))
labels.append('aligned 1:1, NQX=2,3u')
NX.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY.append(1)

# # LinearMapping(1.0), NQX = 8, Qord = 2, quadType = Gauss-Legendre
# E_2.append(np.array([3.44959790e-01, 6.12602939e-03, 1.49298664e-03, 2.60175830e-04,
#        4.45884164e-05]))
# E_inf.append(np.array([1.14265230e+00, 1.90433565e-02, 6.52609015e-03, 1.44858252e-03,
#        2.58464568e-04]))
# labels.append('aligned 1:4, NQX=8,2g')
# NX.append(np.array([  4,   8,  16,  32,  64]))
# NY.append(4)

# # LinearMapping(1.0), NQX = 2, Qord = 2, quadType = Gauss-Legendre
# E_2.append(np.array([1.66169594e+00, 1.69620186e-02, 2.41096928e-03, 4.97752447e-04,
#        1.01626603e-04]))
# E_inf.append(np.array([5.82085945e+00, 7.65940483e-02, 1.02204636e-02, 2.52891644e-03,
#        6.49573968e-04]))
# labels.append('aligned 1:4, NQX=2,2g')
# NX.append(np.array([  4,   8,  16,  32,  64]))
# NY.append(4)

# # LinearMapping(1.0), NQX = 2, Qord = 3, quadType = Gauss-Legendre
# E_2.append(np.array([4.41959593e+00, 1.68091535e-02, 1.73554458e-03, 4.21024560e-04,
#        8.01677881e-05]))
# E_inf.append(np.array([1.75904131e+01, 6.45119958e-02, 8.35585185e-03, 2.11284403e-03,
#        4.27901017e-04]))
# labels.append('aligned 1:4, NQX=2,3g')
# NX.append(np.array([  4,   8,  16,  32,  64]))
# NY.append(4)

# # LinearMapping(1.0), NQX = 3, Qord = 3, quadType = Gauss-Legendre
# E_2.append(np.array([3.51824312e-01, 5.54896748e-03, 1.33837855e-03, 2.30668059e-04,
#        3.79359402e-05]))
# E_inf.append(np.array([1.69869701e+00, 1.92814415e-02, 5.50023563e-03, 1.47238252e-03,
#        2.29241331e-04]))
# labels.append('aligned 1:4, NQX=3,3g')
# NX.append(np.array([  4,   8,  16,  32,  64]))
# NY.append(4)

# # LinearMapping(1.0), NQX = 1, Qord = 4, quadType = Gauss-Legendre
# E_2.append(np.array([1.09973557e+00, 6.02082923e-03, 1.28066069e-03, 1.99170827e-04,
#        3.19313391e-05]))
# E_inf.append(np.array([3.20258949e+00, 1.87350009e-02, 5.53445403e-03, 1.09845947e-03,
#        1.71007950e-04]))
# labels.append('aligned 1:4, NQX=1,4g')
# NX.append(np.array([  4,   8,  16,  32,  64]))
# NY.append(4)

# LinearMapping(1.0), NQX = 1, Qord = 4, quadType = uniform
E_2.append(np.array([9.79455032e-01, 5.40716549e-03, 1.27020551e-03, 1.80888643e-04,
        2.57871979e-05, 3.30007431e-06]))
E_inf.append(np.array([2.81602177e+00, 1.75729625e-02, 5.68631075e-03, 1.13147280e-03,
        1.65057532e-04, 2.42178172e-05]))
labels.append('aligned 1:4, NQX=1,4u')
NX.append(np.array([  4,   8,  16,  32,  64, 128]))
NY.append(4)

# LinearMapping(1.0), NQX = 1, Qord = 4, quadType = uniform
E_2.append(np.array([4.82169629e-02, 1.71950170e-03, 2.92268704e-04, 4.33212514e-05,
       6.52183947e-06]))
E_inf.append(np.array([1.84738650e-01, 6.73128239e-03, 1.46019883e-03, 1.98598135e-04,
       3.82148971e-05]))
labels.append('aligned 1:8, NQX=1,4u')
NX.append(np.array([ 4,  8, 16, 32, 64]))
NY.append(8)

# LinearMapping(1.0), NQX = 1, Qord = 4, quadType = uniform
E_2.append(np.array([3.13801661e-02, 1.48245200e-03, 2.70289858e-04, 3.14147767e-05]))
E_inf.append(np.array([8.62556909e-02, 6.19591903e-03, 1.46127737e-03, 1.77080847e-04]))
labels.append('aligned 1:16, NQX=1,4u')
NX.append(np.array([ 4,  8, 16, 32]))
NY.append(16)

# # LinearMapping(1.0), NQX = 1, Qord = 5, quadType = Gauss-Legendre
# E_2.append(np.array([7.17676847e-01, 6.14319887e-03, 1.25078620e-03, 1.78853126e-04,
#        2.49834358e-05]))
# E_inf.append(np.array([1.93777258e+00, 2.30865131e-02, 5.82127075e-03, 1.09375224e-03,
#        1.49534970e-04]))
# labels.append('aligned 1:4, NQX=1,5g')
# NX.append(np.array([  4,   8,  16,  32,  64]))
# NY.append(4)

# # LinearMapping(1.0), NQX = 1, Qord = 5, quadType = uniform
# E_2.append(np.array([4.86802211e-01, 5.18244268e-03, 1.21784540e-03, 1.70744332e-04,
#        2.30191779e-05]))
# E_inf.append(np.array([2.07768700e+00, 1.99852840e-02, 5.45610152e-03, 1.06844313e-03,
#        1.29661185e-04]))
# labels.append('aligned 1:4, NQX=1,5u')
# NX.append(np.array([  4,   8,  16,  32,  64]))
# NY.append(4)

# # LinearMapping(1.0), NQX = 1, Qord = 5, quadType = uniform
# E_2.append(np.array([3.88550175e-02, 1.68499191e-03, 2.88484017e-04, 4.03326149e-05,
#         5.61124195e-06]))
# E_inf.append(np.array([1.21179301e-01, 7.00808864e-03, 1.29588412e-03, 2.09248923e-04,
#         3.70929167e-05]))
# labels.append('aligned 1:8, NQX=1,5u')
# NX.append(np.array([ 4,  8, 16, 32, 64]))
# NY.append(8)

# # LinearMapping(1.0), NQX = 1, Qord = 5, quadType = uniform
# E_2.append(np.array([2.77648961e-02, 1.42369230e-03, 2.67420295e-04, 2.99999984e-05]))
# E_inf.append(np.array([7.53332438e-02, 5.77539634e-03, 1.37215238e-03, 1.69496502e-04]))
# labels.append('aligned 1:16, NQX=1,5u')
# NX.append(np.array([ 4,  8, 16, 32]))
# NY.append(16)



##### Begin Plotting Routines #####

solid_linewidth = 1.25
dashed_linewidth = 1.0

plt.rc('markers', fillstyle='full')
plt.rc('lines', markersize=5.0)
plt.rc('pdf', fonttype=42)
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# fontsize : int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
# plt.rc('font', size='small')
plt.rc('legend', fontsize='small')
# plt.rc('axes', titlesize='medium', labelsize='medium')
# plt.rc('xtick', labelsize='small')
# plt.rc('ytick', labelsize='small')
# plt.rc('figure', titlesize='large')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
blue = '#1f77b4'
orange = '#ff7f0e'
green = '#2ca02c'
red = '#d62728'
purple = '#9467bd'
brown = '#8c564b'
pink = '#e377c2'
grey = '#7f7f7f'
yellow = '#bcbd22'
cyan = '#17becf'
black = '#000000'

if len(E_2) == 1:
    cycler = plt.cycler(color=[black], marker=['d'])
elif len(E_2) < 4: # 2 and 3
    cycler = plt.cycler(color=[blue, red, black], marker=['o', 's', 'd'])
else: # 4 or more
    cycler = plt.cycler(color=[blue, red, orange, black, green] + colors[4:],
        marker=['o', 's', '^', 'd', 'x', '+', 'v', '<', '>', '*', 'p'])

##### for double plot #####
fig = plt.figure(figsize=(7.75, 3))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
axL1, axR1 = fig.subplots(1, 2)

# ##### for single plot #####
# fig = plt.figure(figsize=(3.875, 3))
# # fig.subplots_adjust(left=0.2, right=0.85)
# fig.subplots_adjust(left=0.2)
# axL1 = fig.subplots(1, 1)

# ##### plot solution at right, requires a sim object from test #####
# fig = plt.figure(figsize=(7.75, 3))
# fig.subplots_adjust(hspace=0.5, wspace=0.3)
# axL1, axR1 = fig.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.1]})
# sim.generatePlottingPoints(nx=2, ny=2)
# sim.computePlottingSolution()
# field = axR1.tripcolor(sim.X, sim.Y, sim.U, shading='gouraud', vmin=-1, vmax=1)
# cbar = fig.colorbar(field, ax=axR1)
# cbar.set_ticks(np.linspace(-1,1,5))
# cbar.set_label(r'$u(x,y)$', rotation=0, labelpad=10)
# x = np.linspace(0, sim.nodeX[-1], 100)
# axR1.plot(x, [sim.mapping(np.array([[0, float(yi)]]), i) for i in x], 'k')
# axR1.margins(0,0)
# axR1.set_xticks(np.linspace(0, f.xmax, 6))
# axR1.set_xlabel(r'$x$')
# axR1.set_ylabel(r'$y$', rotation=0, labelpad=10)


axL1.set_prop_cycle(cycler)
N = []
inds = []
for i, error in enumerate(E_2):
    N.append(np.log2(NY[i]*NX[i]**2).astype('int'))
    inds.append(N[i] >= 4)
    axL1.semilogy(N[i][inds[i]], error[inds[i]]/(2*np.pi), label=labels[i],
                  linewidth=solid_linewidth)
# axL1.minorticks_off()
Nmin = min([min(N[i][inds[i]]) for i in range(len(N))])
Nmax = max([max(N[i][inds[i]]) for i in range(len(N))])
Nstep = 2
axL1.set_xticks(np.linspace(Nmin, Nmax, (Nmax - Nmin)//Nstep + 1))
axL1.set_xlabel(r'$\log_2(N_xN_y)$')
axL1.set_ylabel(r'$|E_2|$', rotation=0, labelpad=10)
axL1.legend(loc='lower left')
xlim = axL1.get_xlim()

axR1.set_prop_cycle(cycler)
axR1.axhline(3, linestyle=':', color=black, label='3rd order', zorder=0,
             linewidth=dashed_linewidth)
axR1.axhline(4, linestyle=':', color=black, label='4th order', zorder=0,
             linewidth=dashed_linewidth)
for i, error in enumerate(E_2):
    logE = np.log(error[inds[i]])
    logN = np.log(NX[i][inds[i]])
    order = (logE[:-1] - logE[1:])/(logN[1:] - logN[:-1])
    intraN = 0.5 * (N[i][inds[i]][:-1] + N[i][inds[i]][1:])
    axR1.plot(intraN, order, linestyle=':', label=labels[i],
              linewidth=dashed_linewidth)
axR1.set_xticks(np.linspace(Nmin, Nmax, (Nmax - Nmin)//Nstep + 1))
axR1.set_xlim(xlim)
axR1.set_xlabel(r'$\log_2(N_xN_y)$')
axR1.set_ylabel(r'Intra-step Order of Convergence')
ordb = 1
ordt = 5
ordstep = 1
axR1.set_ylim(ordb, ordt)
axR1.set_yticks(np.linspace(ordb, ordt, int((ordt - ordb)/ordstep) + 1))
lines, labels = axR1.get_legend_handles_labels()
axR1.legend(lines[2:], labels[2:], loc='lower right')

# fig.savefig("boundary_conv.pdf", bbox_inches = 'tight', pad_inches = 0)
