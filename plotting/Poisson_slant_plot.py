# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:46:29 2021

@author: Samuel A. Maloney
"""

import numpy as np
import matplotlib.pyplot as plt

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


E_2 = []
E_inf = []
t_setup = []
t_solve = []
labels = []
NX = []
NY = []


# For all of the below, unless otherwise noted
# points on axes constrained via Lagrange multipliers
# f(x,y) = 0.5*sin(n*(2pi*y - 2pi*x))*(1 + sin(2pi*y))
# n = 8
# Omega = (0,1) X (0,1)

# px = 0.1, py = 0.1, seed = 42
# basis = quadratic
# boundary = PeriodicBoundary(support = [2.5 2.5])
# NQY = 1*NY
# massLumping = False


# ##### kernel = quintic #####
# ##### VCI: VC2-C (whole domain) using ssqr.min2norm #####
# StraightMapping(), NQX = 2, Qord = 3, quadType = uniform
E_2.append(np.array([2.18313644e-01, 3.18109559e-01, 1.19889058e-01, 1.87964312e-02,
       1.80604660e-03, 2.21082526e-04, 2.94829128e-05]))
E_inf.append(np.array([5.15972736e-01, 1.06550970e+00, 4.05288260e-01, 6.15180906e-02,
       8.51598742e-03, 1.37924482e-03, 2.06605124e-04]))
labels.append(r'unaligned \ratio{1}{1}')
NX.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY.append(1)

# LinearMapping(1.0), NQX = 2, Qord = 3, quadType = uniform
E_2.append(np.array([2.51847989e-01, 4.26166739e-01, 1.19348790e-01, 8.91487179e-03,
        1.25666846e-03, 1.64004712e-04, 2.10771717e-05]))
E_inf.append(np.array([7.61508486e-01, 1.28358217e+00, 4.64935934e-01, 4.74539785e-02,
        6.77682388e-03, 1.15004246e-03, 1.41868404e-04]))
labels.append(r'aligned \ratio{1}{1}')
NX.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY.append(1)

# LinearMapping(1.0), NQX = 2, Qord = 3, quadType = uniform
E_2.append(np.array([2.89082478e-01, 1.20800607e-02, 1.18841530e-03, 1.48747900e-04,
       2.09448621e-05, 3.49088736e-06]))
E_inf.append(np.array([9.34180649e-01, 3.85910014e-02, 6.89805182e-03, 9.20515322e-04,
       1.41250955e-04, 2.16375626e-05]))
labels.append(r'aligned \ratio{1}{4}')
NX.append(np.array([  4,   8,  16,  32,  64, 128]))
NY.append(4)

# LinearMapping(1.0), NQX = 2, Qord = 3, quadType = uniform
E_2.append(np.array([1.05412507e-01, 1.74048543e-03, 2.07302524e-04, 3.37724196e-05,
        5.40016889e-06]))
E_inf.append(np.array([2.67348479e-01, 8.12251050e-03, 8.77554252e-04, 1.74406318e-04,
        3.12707558e-05]))
labels.append(r'aligned \ratio{1}{8}')
NX.append(np.array([ 4,  8, 16, 32, 64]))
NY.append(8)

# LinearMapping(1.0), NQX = 2, Qord = 3, quadType = uniform
E_2.append(np.array([2.80956799e-02, 1.19181622e-03, 1.83449800e-04, 2.50872773e-05,
        3.20576481e-06]))
E_inf.append(np.array([9.12424357e-02, 3.77618428e-03, 1.09074034e-03, 1.09796278e-04,
        1.42691934e-05]))
labels.append(r'aligned \ratio{1}{16}')
NX.append(np.array([ 4,  8, 16, 32, 64]))
NY.append(16)
# used LSQR for final NX=64 data point for 1:16 ratio, took ~45 hours


# ##### kernel = cubic #####
# ##### VCI: VC2-C (whole domain) using ssqr.min2norm #####

# # # StraightMapping(), NQX = 2, Qord = 3, quadType = Gauss-Legendre
# # E_2.append(np.array([2.34448055e+00, 6.57188630e-01, 2.70008304e-01, 1.82040318e-02,
# #        2.56882481e-03, 3.76863982e-04, 6.23491998e-05]))
# # E_inf.append(np.array([7.00554879e+00, 1.94212399e+00, 8.76743557e-01, 1.19757763e-01,
# #        1.47687995e-02, 2.35242704e-03, 4.63392782e-04]))
# # labels.append(r'unaligned \ratio{1}{1}, NQX=2,3g')
# # NX.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# # NY.append(1)

# # # LinearMapping(1.0), NQX = 2, Qord = 3, quadType = Gauss-Legendre
# # E_2.append(np.)
# # E_inf.append(np.)
# # labels.append(r'aligned \ratio{1}{1}, NQX=2,3g')
# # NX.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# # NY.append(1)

# # StraightMapping(), NQX = 2, Qord = 3, quadType = uniform
# E_2.append(np.array([1.49748386e+00, 4.13366596e-01, 2.43554769e-01, 1.22880284e-02,
#         2.34931415e-03, 3.73777518e-04, 5.68981496e-05]))
# E_inf.append(np.array([3.66002239e+00, 1.45171538e+00, 8.41521212e-01, 6.46384067e-02,
#         1.23272660e-02, 2.46645256e-03, 4.04293772e-04]))
# labels.append(r'unaligned \ratio{1}{1}')
# NX.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY.append(1)

# # LinearMapping(1.0), NQX = 2, Qord = 3, quadType = uniform
# E_2.append(np.array([5.58812015e-01, 4.40555489e-01, 2.27694333e-01, 5.63276224e-03,
#         1.30464751e-03, 1.93062275e-04, 2.43904087e-05]))
# E_inf.append(np.array([1.29977295e+00, 1.61485914e+00, 8.02708986e-01, 2.33030391e-02,
#         8.53413834e-03, 1.14301712e-03, 1.56013312e-04]))
# labels.append(r'aligned \ratio{1}{1}')
# NX.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY.append(1)

# # LinearMapping(1.0), NQX = 2, Qord = 3, quadType = uniform
# E_2.append(np.array([8.32725482e-01, 6.83975487e-03, 1.41657777e-03, 2.29692308e-04,
#         4.15332850e-05, 7.80225451e-06]))
# E_inf.append(np.array([1.97781921e+00, 2.26051146e-02, 7.19082036e-03, 1.42557487e-03,
#         2.51456921e-04, 5.19011570e-05]))
# labels.append(r'aligned \ratio{1}{4}')
# NX.append(np.array([  4,   8,  16,  32,  64, 128]))
# NY.append(4)

# # LinearMapping(1.0), NQX = 2, Qord = 3, quadType = uniform
# E_2.append(np.array([1.41905661e-01, 1.76590574e-03, 3.64978620e-04, 6.03782613e-05,
#         1.36267383e-05]))
# E_inf.append(np.array([3.22980550e-01, 8.36416520e-03, 1.53023784e-03, 3.08456060e-04,
#         7.68059481e-05]))
# labels.append(r'aligned \ratio{1}{8}')
# NX.append(np.array([ 4,  8, 16, 32, 64]))
# NY.append(8)

# # # LinearMapping(1.0), NQX = 2, Qord = 3, quadType = uniform
# # E_2.append(np.array([6.13374407e-02, 1.51427170e-03, 2.79576324e-04, 3.60091818e-05]))
# # E_inf.append(np.array([1.93650006e-01, 6.10881333e-03, 1.38417292e-03, 1.54424495e-04]))
# # labels.append(r'aligned \ratio{1}{16}')
# # NX.append(np.array([ 4,  8, 16, 32]))
# # NY.append(16)

# ##### VCI: VC2-C (whole domain) using scipy.sparse.linalg.lsqr #####
# # LinearMapping(1.0), NQX = 2, Qord = 3, quadType = uniform
# E_2.append(np.array([6.14194464e-02, 1.52709273e-03, 2.76761615e-04, 3.63961630e-05,
#         5.35343089e-06]))
# E_inf.append(np.array([1.94450397e-01, 6.15439798e-03, 1.37423467e-03, 1.52596360e-04,
#         2.65487868e-05]))
# labels.append(r'aligned \ratio{1}{16}')
# NX.append(np.array([ 4,  8, 16, 32, 64]))
# NY.append(16)



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

# ##### for double plot #####
# fig = plt.figure(figsize=(7.75, 3))
# fig.subplots_adjust(hspace=0.5, wspace=0.5)
# axL1, axR1 = fig.subplots(1, 2)

# ##### for single plot #####
# fig = plt.figure(figsize=(3.875, 3))
# # fig.subplots_adjust(left=0.2, right=0.85)
# fig.subplots_adjust(left=0.2)
# axL1 = fig.subplots(1, 1)

##### for triple plot #####
fig = plt.figure(figsize=(7.75, 6))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
axes = fig.subplot_mosaic('AB;CC')
axL1 = axes['A']
axR1 = axes['B']

f = slantedTestProblem()
nx = ny = 128
X, Y = np.meshgrid((1/nx)*np.arange(nx+1), (1/ny)*np.arange(ny+1))
U = f.solution(np.hstack((X.reshape(-1,1), Y.reshape(-1,1))))*(1/f.umax)
# U = f(np.hstack((X.reshape(-1,1), Y.reshape(-1,1))))
ax = axes['C']
field = ax.tripcolor(X.ravel(), Y.ravel(), U, shading='gouraud',
                     cmap='viridis', vmin=-1, vmax=1)
ax.margins(0,0)
ax.set_xticks(np.linspace(0, 1, 6))
ax.set_yticks(np.linspace(0, 1, 6))
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$', verticalalignment='center', rotation=0, labelpad=13)
ax.set_aspect(1, anchor='C')
cax = ax.inset_axes([1.05, 0, 0.05, 1])
cbar = fig.colorbar(field, ax=ax, cax=cax)
cbar.set_ticks(np.linspace(-1,1,5))
cbar.set_label(r'$u(x,y)$', rotation=0, labelpad=0,
    verticalalignment='center', horizontalalignment='left')


axL1.set_prop_cycle(cycler)
N = []
inds = []
for i, error in enumerate(E_2):
    N.append(np.log2(NY[i]*NX[i]**2).astype('int'))
    inds.append(N[i] >= 4)
    axL1.semilogy(N[i][inds[i]], error[inds[i]]/(2*np.pi), label=labels[i],
                  linewidth=solid_linewidth)
Nmin = min([min(N[i][inds[i]]) for i in range(len(N))])
Nmax = max([max(N[i][inds[i]]) for i in range(len(N))])
Nstep = 2
axL1.set_xticks(np.linspace(Nmin, Nmax, (Nmax - Nmin)//Nstep + 1))
axL1.set_xlabel(r'$\log_2(N_xN_y)$')
axL1.set_ylabel(r'$\norm{u-u^d}$')
axL1.legend(loc='lower left', framealpha=1)
# axL1.legend(loc='upper right', framealpha=1)
# ylim = axL1.get_ylim()
# axL1.set_ylim((1e-5, 1e-3))
axL1.set_yticks((1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-3, 1e-2, 1e-1))
# axL1.set_yticks((1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
#                 (r'$10^{-7}$','',r'$10^{-5}$','',r'$10^{-3}$','',r'$10^{-1}$'))
# axL1.set_yticks((1e-7, 1e-5, 1e-3, 1e-1))
# axL1.minorticks_on()
# axL1.yaxis.get_minor_locator().set_params(numticks=99, subs=[.2, .4, .6, .8])
axL1.yaxis.get_minor_locator().set_params(numticks=99, subs='auto')
xlim = axL1.get_xlim()

axR1.set_prop_cycle(cycler)
axR1.axhline(3, linestyle=':', color=black, label='3rd order', zorder=0,
             linewidth=dashed_linewidth)
# axR1.axhline(2, linestyle=':', color=black, label='2nd order', zorder=0,
#               linewidth=dashed_linewidth)
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
ordb = 1.0
ordt = 5.0
ordstep = 1
axR1.set_ylim(ordb, ordt)
axR1.set_yticks(np.linspace(ordb, ordt, int((ordt - ordb)/ordstep) + 1))
# axR1.set_yticks(np.linspace(2, 4, 5))
lines, labels = axR1.get_legend_handles_labels()
# axR1.legend(lines[2:], labels[2:], loc='upper left', framealpha=1)

# fig.savefig("fcimls_slant_conv.pdf", bbox_inches = 'tight', pad_inches = 0)
