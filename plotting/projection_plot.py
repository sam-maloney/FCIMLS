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
# seed = 42
# basis = quadratic, kernel = cubic
# boundary = PeriodicBoundary(support = [2.5 2.5])
# VCI: None using None
# NQY = 1*NY, massLumping = False

# NQX = 1, Qord = 3, quadType = Gauss-Legendre

# StraightMapping()
E_2_L.append(np.array([4.65999900e-02, 5.94109035e-05, 1.43721487e-05, 1.09886322e-06,
        7.19851610e-08, 4.55145122e-09, 2.85286805e-10]))
E_inf_L.append(np.array([6.11559661e-02, 1.18821807e-04, 2.87442974e-05, 2.19772644e-06,
        1.43970328e-07, 9.10290621e-09, 5.70569592e-10]))
labels_L.append('uniform str, Q3g')
NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_L.append(1)

# SinusoidalMapping(0.2, -0.25, 1.0)
E_2_L.append(np.array([8.32524772e-02, 7.26433188e-03, 1.61502000e-04, 1.53484573e-05,
        1.32226095e-06, 9.04133067e-08, 5.78855878e-09]))
E_inf_L.append(np.array([1.61216413e-01, 1.35052380e-02, 3.31961002e-04, 3.91859526e-05,
        3.52978262e-06, 2.44894619e-07, 1.57024289e-08]))
labels_L.append('uniform sin, Q3g')
NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_L.append(1)

# StraightMapping()
E_2_L.append(np.array([4.20261832e-02, 2.81755319e-03, 4.10473462e-04, 6.25776947e-05,
       7.79852079e-06, 9.80107779e-07, 1.28408789e-07]))
E_inf_L.append(np.array([7.68510229e-02, 6.25620176e-03, 1.23054151e-03, 2.28754608e-04,
       3.38327122e-05, 4.41788719e-06, 6.28913255e-07]))
labels_L.append(r'10\% pert.  str, Q3g')
NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_L.append(1)

# SinusoidalMapping(0.2, -0.25, 1.0)
E_2_L.append(np.array([9.75924522e-02, 9.10705558e-03, 1.55401755e-03, 2.46524649e-04,
       3.17185188e-05, 4.09622569e-06, 5.65889782e-07]))
E_inf_L.append(np.array([1.87602041e-01, 2.55664716e-02, 5.10593428e-03, 8.68615593e-04,
       1.42100620e-04, 2.15754215e-05, 3.02108332e-06]))
labels_L.append(r'10\% pert.  sin, Q3g')
NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_L.append(1)


# # NQX = 1, Qord = 3, quadType = uniform

# # SinusoidalMapping(0.2, -0.25, 1.0)
# E_2_L.append(np.array([9.72991529e-02, 8.92840504e-03, 1.56263974e-03, 2.47731920e-04,
#        3.22430995e-05, 4.14427815e-06, 5.73999051e-07]))
# E_inf_L.append(np.array([1.80413889e-01, 2.58976844e-02, 5.07037162e-03, 8.70739581e-04,
#        1.41066198e-04, 2.16990192e-05, 2.92921867e-06]))
# labels_L.append(r'10\% pert.  sin, Q3u')
# NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_L.append(1)


# # NQX = 1, Qord = 4, quadType = uniform

# # SinusoidalMapping(0.2, -0.25, 1.0)
# E_2_L.append(np.array([8.14531131e-02, 7.05257545e-03, 1.49770402e-04, 1.72810125e-05,
#        1.43999113e-06, 9.76399920e-08, 6.23948541e-09]))
# E_inf_L.append(np.array([1.57443654e-01, 1.30741566e-02, 3.14908437e-04, 4.22169159e-05,
#        3.69630590e-06, 2.52765809e-07, 1.61459307e-08]))
# labels_L.append('uniform sin, Q4u')
# NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_L.append(1)

# # SinusoidalMapping(0.2, -0.25, 1.0)
# E_2_L.append(np.array([9.58371782e-02, 8.91697571e-03, 1.57000081e-03, 2.48177454e-04,
#         3.22469015e-05, 4.14348923e-06, 5.75092035e-07]))
# E_inf_L.append(np.array([1.80747303e-01, 2.55861767e-02, 5.16201464e-03, 8.71502587e-04,
#         1.40252799e-04, 2.17092059e-05, 2.94970639e-06]))
# labels_L.append(r'10\% pert.  sin, Q4u')
# NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_L.append(1)



##### Dirichlet BCs #####
# f(x,y) = x*sin(2pi*n(y - a*x^2 - b*x))
# Omega = (0,1) X (0,1)
# n = 3, a = (1 - xmax*b)/xmax^2 = 0.95, b = 0.05
# QuadraticMapping(0.95, 0.05)
# seed = 42
# basis = quadratic, kernel = cubic
# NQX = 1, NQY = 1*NY, Qord = 3
# massLumping = False, quadType = Gauss-Legendre
# VCI: None using None

# boundary = DirichletBoundary(support = [3. 3.], NDX=2)
E_2_R.append(np.array([2.35142926e-01, 1.82443837e-02, 1.36899798e-03, 1.71059637e-04,
       1.57778638e-05, 1.41319139e-06, 1.25336452e-07]))
E_inf_R.append(np.array([8.92557643e-01, 1.00563790e-01, 9.99520056e-03, 1.47779970e-03,
       1.90233726e-04, 3.30875242e-05, 4.60217619e-06]))
labels_R.append('uniform 1:1, NDX=2')
NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_R.append(1)

# boundary = DirichletBoundary(support = [3. 3.], NDX = 3)
E_2_R.append(np.array([1.63872925e-01, 1.81152275e-02, 1.07937196e-03, 1.46783148e-04,
       1.17115318e-05, 9.77800513e-07, 8.45481427e-08]))
E_inf_R.append(np.array([5.72562375e-01, 1.07986010e-01, 6.42286228e-03, 1.16829344e-03,
       1.15581208e-04, 1.32643729e-05, 1.72208848e-06]))
labels_R.append('uniform 1:1, NDX=3')
NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_R.append(1)


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
    cycler = plt.cycler(color=[black, blue, red, orange, green] + colors[4:], 
        marker=['d', 'o', 's', '^', 'x', '+', 'v', '<', '>', '*', 'p'])

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
    fillstyle = 'full'
    # if i < len(E_2_R)/2:
    #     fillstyle = 'full'
    # else:
    #     fillstyle = 'none'
    axR1.semilogy(N[i][inds[i]], error[inds[i]], label=labels_R[i],
                  linewidth=solid_linewidth, fillstyle=fillstyle)

    logE = np.log(error[inds[i]])
    logN = np.log(NX_R[i][inds[i]])
    order = (logE[:-1] - logE[1:])/(logN[1:] - logN[:-1])
    intraN = 0.5 * (N[i][inds[i]][:-1] + N[i][inds[i]][1:])
    axR2.plot(intraN, order, linestyle=':', label=labels_R[i],
              linewidth=dashed_linewidth, fillstyle=fillstyle)
axR2.axhline(3, linestyle=':', color=black, label='3rd order', zorder=0,
             linewidth=dashed_linewidth)
axR2.axhline(4, linestyle=':', color=black, label='4th order', zorder=0,
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
