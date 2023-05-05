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

##### Linear FCIFEM

# For all of the below, unless otherwise noted
# Left and bottom borders and centre point constrained
# f(x,y) = 0.5*sin(n*(2pi*y - 2pi*x))*(1 + sin(2pi*y))
# n = 8
# Omega = (0,1) X (0,1)
# Periodic BCs with VCI-C (slice-by-slice)
# NQY = NY, quadType = 'gauss', massLumping = False
# px = py = 0.1, seed = 42, Qord = 2

# # StraightMapping()
# # NQX = 2
# E_2.append(np.array([5.23788300e+00, 3.88013543e-01, 1.18140381e+00, 1.38561964e-01,
#        2.52995617e-02, 5.75341209e-03, 1.37187530e-03]))
# E_inf.append(np.array([1.29122915e+01, 1.14943804e+00, 3.54597385e+00, 6.32682732e-01,
#        1.18120233e-01, 2.80396183e-02, 6.33708214e-03]))
# labels.append('unaligned 1:1')
# NX.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY.append(1)

# LinearMapping(1.0)
# NQX = 2
E_2.append(np.array([2.79491854e+00, 2.29866277e+00, 8.57252448e-01, 5.38464637e-02,
       1.04973528e-02, 2.40780701e-03, 5.59523070e-04]))
E_inf.append(np.array([7.65053640e+00, 8.22621535e+00, 3.00392425e+00, 2.40594243e-01,
       6.56883782e-02, 1.44502250e-02, 4.04024302e-03]))
labels.append('aligned 1:1')
NX.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY.append(1)

# LinearMapping(1.0)
# NQX = 8
E_2.append(np.array([3.01824203e-01, 3.89226408e-01, 1.05572199e-01, 2.24640453e-02,
       4.45034078e-03, 1.09990137e-03, 2.34753604e-04]))
E_inf.append(np.array([8.10010193e-01, 1.23491188e+00, 4.87011097e-01, 1.21294623e-01,
       2.06432667e-02, 7.56082282e-03, 2.21205443e-03]))
labels.append('aligned 1:4')
NX.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
NY.append(4)

# LinearMapping(1.0)
# NQX = 16
E_2.append(np.array([3.85639237e-01, 1.38313698e-01, 2.36056443e-02, 7.12212284e-03,
       1.49412789e-03, 3.79996990e-04]))
E_inf.append(np.array([9.87808995e-01, 4.77594313e-01, 9.52832583e-02, 3.48203712e-02,
       7.23733090e-03, 2.10659547e-03]))
labels.append('aligned 1:8')
NX.append(np.array([ 2,  4,  8, 16, 32, 64]))
NY.append(8)



##### Quadratic MLS

# For all of the below, unless otherwise noted
# points on axes constrained via Lagrange multipliers
# f(x,y) = 0.5*sin(n*(2pi*y - 2pi*x))*(1 + sin(2pi*y))
# n = 8
# Omega = (0,1) X (0,1)

# px = 0.1, py = 0.1, seed = 42
# basis = quadratic, kernel = cubic
# boundary = PeriodicBoundary(support = [2.5 2.5])
# NQY = 1*NY
# massLumping = False, quadType = uniform

# VCI: VC2-C (whole domain) using ssqr.min2norm


# # StraightMapping(), NQX = 2, Qord = 3
# E_2.append(np.array([3.66500704e+00, 1.79102106e+00, 2.44918893e-01, 1.60849314e-02,
#        2.47328472e-03, 4.55351099e-04, 8.58986774e-05]))
# E_inf.append(np.array([1.15697486e+01, 6.40099115e+00, 8.48972481e-01, 7.51296999e-02,
#        1.13340149e-02, 2.39105291e-03, 3.65807720e-04]))
# labels.append(r'unaligned \ratio{1}{1}')
# NX.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY.append(1)

# LinearMapping(1.0), NQX = 2, Qord = 3
E_2.append(np.array([4.50816969e+00, 4.35767178e+00, 2.27755332e-01, 5.66164282e-03,
       1.30324143e-03, 1.92936997e-04, 2.43886935e-05]))
E_inf.append(np.array([1.12285705e+01, 1.16367501e+01, 8.02927946e-01, 2.25277766e-02,
       8.54735061e-03, 1.13792309e-03, 1.56774456e-04]))
labels.append(r'aligned \ratio{1}{1}')
NX.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY.append(1)

# LinearMapping(1.0), NQX = 1, Qord = 4
E_2.append(np.array([9.79455032e-01, 5.40716549e-03, 1.27020551e-03, 1.80888643e-04,
        2.57871979e-05, 3.30007431e-06]))
E_inf.append(np.array([2.81602177e+00, 1.75729625e-02, 5.68631075e-03, 1.13147280e-03,
        1.65057532e-04, 2.42178172e-05]))
labels.append(r'aligned \ratio{1}{4}')
NX.append(np.array([  4,   8,  16,  32,  64, 128]))
NY.append(4)

# LinearMapping(1.0), NQX = 1, Qord = 4
E_2.append(np.array([4.82169629e-02, 1.71950170e-03, 2.92268704e-04, 4.33212514e-05,
       6.52183947e-06]))
E_inf.append(np.array([1.84738650e-01, 6.73128239e-03, 1.46019883e-03, 1.98598135e-04,
       3.82148971e-05]))
labels.append(r'aligned \ratio{1}{8}')
NX.append(np.array([ 4,  8, 16, 32, 64]))
NY.append(8)



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

nSeries = len(E_2)//2

if nSeries == 1:
    cycler = plt.cycler(color=[black], marker=['d'])
elif nSeries == 2:
    cycler = plt.cycler(color=[blue, red], marker=['o', 's'])
elif nSeries == 3:
    cycler = plt.cycler(color=[blue, red, black], marker=['o', 's', 'd'])
elif nSeries == 4:
    cycler = plt.cycler(color=[blue, red, orange, black],
                        marker=['o', 's', '^', 'd'])
else:
    pass # should never be more than 4

##### for double plot #####
fig = plt.figure(figsize=(7.75, 3))
# fig.subplots_adjust(hspace=0.5, wspace=0.5)
# axL1, axR1 = fig.subplots(1, 2)


##### plot solution at right #####
fig.subplots_adjust(hspace=0.5, wspace=0.3)
axL1, axR1 = fig.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.1]})

f = slantedTestProblem()
nx = 64
ny = nx
X, Y = np.meshgrid((1/nx)*np.arange(nx+1), (1/ny)*np.arange(ny+1))
U = f.solution(np.hstack((X.reshape(-1,1), Y.reshape(-1,1))))*(1/f.umax)
field = axR1.tripcolor(X.ravel(), Y.ravel(), U, shading='gouraud', vmin=-1, vmax=1)
cbar = fig.colorbar(field, ax=axR1)
cbar.set_ticks(np.linspace(-1,1,5))
cbar.set_label(r'$u(x,y)$', rotation=0, labelpad=10)
# x = np.linspace(0, sim.nodeX[-1], 100)
# axR1.plot(x, [sim.mapping(np.array([[0, float(yi)]]), i) for i in x], 'k')
axR1.margins(0,0)
axR1.set_xticks(np.linspace(0, 1, 6))
axR1.set_xlabel(r'$x$')
axR1.set_ylabel(r'$y$', rotation=0, labelpad=10)


axL1.set_prop_cycle(cycler)
N = []
inds = []
handles = []
for i, error in enumerate(E_2):
    N.append(np.log2(NY[i]*NX[i]**2).astype('int'))
    inds.append(N[i] >= 4)
    if i < nSeries:
        fillstyle = 'none'
    else:
        fillstyle = 'full'
    
    handles.append(axL1.semilogy(N[i][inds[i]], error[inds[i]]/(2*np.pi),
        label=labels[i], linewidth=solid_linewidth, fillstyle=fillstyle)[0])
# axL1.minorticks_off()
Nmin = min([min(N[i][inds[i]]) for i in range(len(N))])
Nmax = max([max(N[i][inds[i]]) for i in range(len(N))])
Nstep = 2
axL1.set_xticks(np.linspace(Nmin, Nmax, (Nmax - Nmin)//Nstep + 1))
axL1.set_xlabel(r'$\log_2(N_xN_y)$')
axL1.set_ylabel(r'$|E_2|$', rotation=0, labelpad=10)
legend1 = axL1.legend(handles=handles[nSeries:], loc='lower left')
legend2 = axL1.legend(handles[::nSeries], ('linear FCIFEM', 'quadratic FCIMLS'), loc='upper right')
axL1.add_artist(legend1)
xlim = axL1.get_xlim()


# ##### plot Intra-step Orders of Convergence at right #####

# axR1.set_prop_cycle(cycler)
# axR1.axhline(3, linestyle=':', color=black, label='3rd order', zorder=0,
#              linewidth=dashed_linewidth)
# axR1.axhline(4, linestyle=':', color=black, label='4th order', zorder=0,
#              linewidth=dashed_linewidth)
# for i, error in enumerate(E_2):
#     logE = np.log(error[inds[i]])
#     logN = np.log(NX[i][inds[i]])
#     order = (logE[:-1] - logE[1:])/(logN[1:] - logN[:-1])
#     intraN = 0.5 * (N[i][inds[i]][:-1] + N[i][inds[i]][1:])
#     axR1.plot(intraN, order, linestyle=':', label=labels[i],
#               linewidth=dashed_linewidth)
# axR1.set_xticks(np.linspace(Nmin, Nmax, (Nmax - Nmin)//Nstep + 1))
# axR1.set_xlim(xlim)
# axR1.set_xlabel(r'$\log_2(N_xN_y)$')
# axR1.set_ylabel(r'Intra-step Order of Convergence')
# ordb = 1
# ordt = 5
# ordstep = 1
# axR1.set_ylim(ordb, ordt)
# axR1.set_yticks(np.linspace(ordb, ordt, int((ordt - ordb)/ordstep) + 1))
# lines, labels = axR1.get_legend_handles_labels()
# axR1.legend(lines[2:], labels[2:], loc='lower right')

# fig.savefig("slant_comparison_plot.pdf", bbox_inches = 'tight', pad_inches = 0)
