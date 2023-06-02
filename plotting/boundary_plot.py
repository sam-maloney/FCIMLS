# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:46:29 2021

@author: Samuel A. Maloney
"""

import numpy as np
import matplotlib.pyplot as plt

# For all of the below, unless otherwise noted
# xmax = 1.0
# px = 0.1, py = 0.1, seed = 42
# basis = quadratic, kernel = quintic
# boundary = DirichletBoundary(support = [3.2 3.2], NDX = 2)
# NQX = 1, NQY = 1*NY, massLumping = False
# Qord = 4, quadType = uniform
# VCI: VC2 (assumed strain) using None

E_2 = []
E_inf = []
labels = []
NX = []
NY = []


# StraightMapping()
E_2.append(np.array([1.84522804e-01, 2.51310330e-01, 2.06529339e-01, 3.16656127e-02,
       3.12807300e-03, 3.95472192e-04, 4.70262729e-05]))
E_inf.append(np.array([6.65306432e-01, 9.06884376e-01, 1.00818599e+00, 1.83232162e-01,
       2.50157690e-02, 4.12289234e-03, 6.86383559e-04]))
NX.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NX.append(np.array([  2,   4,   8,  16,  32,  64, 128, 256]))
labels.append(r'unaligned \ratio{1}{1}')
NY.append(1)

# # StraightMapping()
# E_2.append(np.array([]))
# E_inf.append(np.array([]))
# labels.append(r'unaligned \ratio{1}{4}')
# NX.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY.append(4)

# # StraightMapping()
# E_2.append(np.array([]))
# E_inf.append(np.array([]))
# labels.append(r'unaligned \ratio{4}{1}')
# NX.append(np.array([  8,  16,  32,  64, 128, 256, 512]))
# NY.append(0.25)

# QuadraticMapping(0.95, 0.05)
E_2.append(np.array([1.97548710e-01, 1.93804349e-01, 2.04907566e-02, 2.65789956e-03,
       4.36458020e-04, 5.52925459e-05, 6.75711712e-06]))
E_inf.append(np.array([7.12272003e-01, 8.62563545e-01, 7.92191289e-02, 1.71403059e-02,
       2.30633962e-03, 4.61826169e-04, 5.71112084e-05]))
labels.append(r'aligned \ratio{1}{1}')
NX.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NX.append(np.array([  2,   4,   8,  16,  32,  64, 128, 256]))
NY.append(1)


### These ratios just don't work with this quadratic problem (or doubly-Dirichlet BCs...)

# # QuadraticMapping(0.95, 0.05)
# E_2.append(np.array([]))
# E_inf.append(np.array([]))
# labels.append(r'aligned \ratio{1}{4}')
# NX.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY.append(4)

# # QuadraticMapping(0.95, 0.05)
# E_2.append(np.array([]))
# E_inf.append(np.array([]))
# labels.append(r'aligned \ratio{1}{16}')
# NX.append(np.array([  2,   4,   8,  16,  32,  64]))
# NY.append(16)
# # overall convergence order <> from np.polynomial.polynomial.polyfit(logN, logE, 1)


##### Begin Plotting Routines #####

solid_linewidth = 1.25
dashed_linewidth = 1.0

plt.rc('markers', fillstyle='full')
plt.rc('lines', markersize=5.0)
plt.rc('pdf', fonttype=42)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage[T1]{fontenc}'
                                r'\usepackage[osf,largesc]{newpxtext}'
                                # r'\usepackage[osf,nohelv,largesc]{newpxtext}'
                                r'\usepackage[euler-digits]{eulervm}'
                                # r'\usepackage{eulerpx}'
                                # r'\usepackage[sans]{libertinus}'
                                r'\usepackage{classico}'
                                r'\usepackage{mathtools}'
                                r'\newcommand*{\ratio}[2]{\ensuremath{#1\mathop{:}#2}}'
                                r'\newcommand*{\norm}[1]{\left\lVert#1\right\rVert}'
                                )
plt.rc('font', family='sans-serif')
# plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
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
axR1.axhline(3, linestyle=':', color=black, label='Expected order',
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
ordb = 0
ordt = 4
ordstep = 0.5
axR1.set_ylim(ordb, ordt)
axR1.set_yticks(np.linspace(ordb, ordt, int((ordt - ordb)/ordstep) + 1))
lines, labels = axR1.get_legend_handles_labels()
axR1.legend(lines[1:], labels[1:], loc='lower right')

# fig.savefig("boundary_conv.pdf", bbox_inches = 'tight', pad_inches = 0)
