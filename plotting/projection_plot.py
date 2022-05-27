# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:46:29 2021

@author: Samuel A. Maloney

"""

import numpy as np
import matplotlib.pyplot as plt

E_2_L = []
E_inf_L = []
labels_L = []
NX_L = []
NY_L = []

E_2_R = []
E_inf_R = []
labels_R = []
NX_R = []
NY_R = []

##### Doubly-Periodic BCs #####
# f(x,y) = sin(2pi*x)sin(2pi*y)
# Omega = (0,1) X (0,1)
# NQX = NDX = 1
# Qord = 2

# StraightMapping()
E_2_L.append(np.array([4.59380210e-02, 1.64649412e-04, 1.95053655e-05, 1.38818532e-06,
       8.95326243e-08, 5.65652491e-09, 8.54278227e-10]))
E_inf_L.append(np.array([5.92673088e-02, 3.29298825e-04, 3.90107326e-05, 2.77637123e-06,
       1.79122629e-07, 1.23209279e-08, 1.14972094e-08]))
labels_L.append('uniform str')
NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_L.append(1)

# SinusoidalMapping(0.2, -0.25, 1.0)
E_2_L.append(np.array([6.09539152e-02, 5.15604785e-03, 1.15603474e-04, 1.67386861e-05,
       1.34642736e-06, 9.05235987e-08, 5.98157412e-09]))
E_inf_L.append(np.array([1.13626467e-01, 9.47129774e-03, 2.35050485e-04, 4.17249525e-05,
       3.48851667e-06, 2.41524602e-07, 2.68611869e-08]))
labels_L.append('uniform sin')
NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_L.append(1)

# StraightMapping()
E_2_L.append(np.array([7.55892069e-02, 1.24487786e-02, 1.42816920e-03, 2.15788432e-04,
       2.65232119e-05, 3.24772720e-06, 4.47114092e-07]))
E_inf_L.append(np.array([1.78618805e-01, 3.20571707e-02, 5.52688500e-03, 9.20900929e-04,
       1.47692518e-04, 2.00185268e-05, 2.82010589e-06]))
labels_L.append(r'50\% pert.  str')
NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_L.append(1)

# SinusoidalMapping(0.2, -0.25, 1.0)
E_2_L.append(np.array([2.14811172e-01, 4.45118763e-02, 6.78210435e-03, 8.57406778e-04,
       1.05183926e-04, 1.35815213e-05, 1.97946831e-06]))
E_inf_L.append(np.array([3.59123928e-01, 1.34303209e-01, 2.42143052e-02, 3.43186143e-03,
       4.35285958e-04, 8.56854962e-05, 1.45270726e-05]))
labels_L.append(r'50\% pert.  sin')
NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_L.append(1)

##### Dirichlet BCs #####
# f(x,y) = x*sin(2pi*n(y - a*x^2 - b*x))
# Omega = (0,1) X (0,1)
# n = 3, a = (1 - xmax*b)/xmax^2 = 0.95, b = 0.05
# QuadraticMapping(0.95, 0.05)
# NQX = NDX = 1
# Qord = 2

# E_2_R.append(np.array([3.08346777e-01, 2.32430829e-01, 5.88064927e-02, 1.36571113e-02,
#        3.22287817e-03, 7.82371947e-04, 1.92750314e-04]))
# E_inf_R.append(np.array([6.14369030e-01, 7.48136301e-01, 2.88660465e-01, 9.31112471e-02,
#        2.05478114e-02, 5.18048007e-03, 1.29888727e-03]))
# labels_R.append('uniform 1:1')
# NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_R.append(1)


##### Begin Plotting Routines #####

solid_linewidth = 1.25
dashed_linewidth = 1.0

plt.rc('markers', fillstyle='full')
plt.rc('lines', markersize=5.0, linewidth=solid_linewidth)
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

if len(E_2_L) == 1:
    cycler = plt.cycler(color=[black], marker=['d'])
elif len(E_2_L) == 2:
    cycler = plt.cycler(color=[blue, red], marker=['o', 's'])
else:
    cycler = plt.cycler(color=[blue, red, black], marker=['o', 's', 'd'])

fig = plt.figure(figsize=(7.75, 3))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

axL1, axR1 = fig.subplots(1, 2)
axL2 = axL1.twinx()
axR2 = axR1.twinx()
axL1.set_prop_cycle(cycler)
axL2.set_prop_cycle(cycler)
axR1.set_prop_cycle(cycler)
axR2.set_prop_cycle(cycler)

N = []
inds = []
for i, error in enumerate(E_2_L):
    N.append(np.log2(NY_L[i]*NX_L[i]**2).astype('int'))
    inds.append(N[i] >= 2)
    axL1.semilogy(N[i][inds[i]], error[inds[i]], label=labels_L[i],
                  linewidth=solid_linewidth)

    logE = np.log(error[inds[i]])
    logN = np.log(NX_L[i][inds[i]])
    order = (logE[:-1] - logE[1:])/(logN[1:] - logN[:-1])
    intraN = 0.5 * (N[i][inds[i]][:-1] + N[i][inds[i]][1:])
    axL2.plot(intraN, order, linestyle=':', label=labels_L[i],
              linewidth=dashed_linewidth)
axL2.axhline(3, linestyle=':', color=black, label='3rd order', zorder=0,
             linewidth=dashed_linewidth)
axL2.axhline(4, linestyle=':', color=black, label='4th order', zorder=0,
             linewidth=dashed_linewidth)
# axL1.minorticks_off()
axL1.set_title(r'Doubly-Periodic BCs')
axL1.set_xlabel(r'$\log_2(N_xN_y)$')
axL1.set_ylabel(r'$|E_2|$', rotation=0, labelpad=10)
axL2.set_ylabel(r'Intra-step Order of Convergence')
axL1.legend(loc='lower left')
Nmin = min([min(N[i]) for i in range(len(N))])
Nmax = max([max(N[i]) for i in range(len(N))])
Nstep = 2
axL1.set_xticks(np.linspace(Nmin, Nmax, (Nmax - Nmin)//Nstep + 1))
ordb = 1
ordt = 5
ordstep = 1
axL2.set_ylim(ordb, ordt)
axL2.set_yticks(np.linspace(ordb, ordt, int((ordt - ordb)/ordstep) + 1))

N = []
inds = []
for i, error in enumerate(E_2_R):
    N.append(np.log2(NY_R[i]*NX_R[i]**2).astype('int'))
    inds.append(N[i] >= 2)
    if i < len(E_2_R)/2:
        fillstyle = 'full'
    else:
        fillstyle = 'none'
    axR1.semilogy(N[i][inds[i]], error[inds[i]], label=labels_R[i],
                  linewidth=solid_linewidth, fillstyle=fillstyle)

    logE = np.log(error[inds[i]])
    logN = np.log(NX_R[i][inds[i]])
    order = (logE[:-1] - logE[1:])/(logN[1:] - logN[:-1])
    intraN = 0.5 * (N[i][inds[i]][:-1] + N[i][inds[i]][1:])
    axR2.plot(intraN, order, linestyle=':', label=labels_R[i],
              linewidth=dashed_linewidth, fillstyle=fillstyle)
axR2.axhline(2, linestyle=':', color=black, label='Expected order', zorder=0,
             linewidth=dashed_linewidth)
# axR1.minorticks_off()
axR1.set_title('Dirichlet BCs')
axR1.set_xlabel(r'$\log_2(N_xN_y)$')
axR1.set_ylabel(r'$|E_2|$', rotation=0, labelpad=10)
axR2.set_ylabel(r'Intra-step Order of Convergence')
axR1.legend(loc='upper right')
Nmin = min([min(N[i]) for i in range(len(N))])
Nmax = max([max(N[i]) for i in range(len(N))])
Nstep = 2
axR1.set_xticks(np.linspace(Nmin, Nmax, (Nmax - Nmin)//Nstep + 1))
axR2.set_ylim(ordb, ordt)
axR2.set_yticks(np.linspace(ordb, ordt, int((ordt - ordb)/ordstep) + 1))
axR1.set_ylim(top=2)

# fig.savefig('L2_conv.pdf', bbox_inches='tight', pad_inches=0)
