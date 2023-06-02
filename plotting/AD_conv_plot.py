# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:46:29 2021

@author: Samuel A. Maloney
"""

import numpy as np
import matplotlib.pyplot as plt

# For all of the below, unless otherwise noted
# u0(x,y) = sin(2*pi*x - 2*pi*y)
# Omega = (0,1) X (0,1), Periodic BCs
# dt = 0.01, t_final = 1, theta = 0.7853981633974483
# px = 0.1, py = 0.1, seed = 42
# basis = quadratic, kernel = quintic
# boundary = PeriodicBoundary(support = [2.5 2.5])
# NQX = 2, NQY = 1*NY, massLumping = False
# Qord = 3, quadType = uniform
# VCI: None using None


E_2_L = []
E_inf_L = []
labels_L = []
NX_L = []
NY_L = []

E_2_R = []
E_inf_R = []
delta_u_R = []
labels_R = []
NX_R = []
NY_R = []


# StraightMapping()
E_2_R.append(np.array([6.40658638e-01, 9.63522642e-02, 7.19107541e-03, 5.14234967e-04,
       3.61851800e-05, 4.48960920e-06]))
E_inf_R.append(np.array([9.12161470e-01, 1.46013875e-01, 1.08459182e-02, 1.15292838e-03,
       9.98271619e-05, 1.67526941e-05]))
delta_u_R.append(np.array([ 1.14885474e-03,  3.93789957e-05,  6.51320272e-07, -1.07149677e-08,
       -3.16252743e-11, -1.79817496e-11]))
labels_R.append('unaligned 1:1')
NX_R.append(np.array([  4,   8,  16,  32,  64, 128]))
# NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_R.append(1)

# LinearMapping(1.0)
E_2_R.append(np.array([4.68558574e-01, 1.63537204e-03, 6.05585190e-04, 7.93875103e-05,
       1.92776226e-05, 3.27467283e-06]))
E_inf_R.append(np.array([6.72511603e-01, 3.54229596e-03, 1.58981754e-03, 2.37694566e-04,
       5.58852840e-05, 1.32357881e-05]))
delta_u_R.append(np.array([-1.91220086e-03, -1.59039512e-06, -3.32439562e-07, 4.89534859e-08,
       -1.34220465e-09, -3.06164860e-11]))
labels_R.append('aligned 1:1')
NX_R.append(np.array([  4,   8,  16,  32,  64, 128]))
# NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_R.append(1)

# LinearMapping(1.0)
E_2_R.append(np.array([6.19208922e-03, 5.83689527e-05, 1.02646908e-05, 1.58596198e-06,
       2.71023891e-07]))
E_inf_R.append(np.array([1.10986095e-02, 1.57276681e-04, 3.29208259e-05, 6.98852211e-06,
       1.08436038e-06]))
delta_u_R.append(np.array([3.36459329e-07, 1.64555818e-08, -1.29527376e-09, 2.27280531e-10,
       -3.32858758e-12]))
labels_R.append('aligned 1:4')
NX_R.append(np.array([  4,   8,  16,  32,  64]))
# NX_R.append(np.array([  4,   8,  16,  32,  64, 128]))
NY_R.append(4)

# LinearMapping(1.0)
E_2_R.append(np.array([3.89710186e-04, 1.07041545e-06, 1.40137572e-07, 2.60273718e-08]))
E_inf_R.append(np.array([6.39282500e-04, 4.16906124e-06, 5.74302139e-07, 1.04106719e-07]))
delta_u_R.append(np.array([1.48500588e-07, -4.45943861e-10, 3.05815176e-11, 3.35609972e-13]))
labels_R.append('aligned 1:16')
NX_R.append(np.array([ 4,  8, 16, 32]))
# NX_R.append(np.array([ 4,  8, 16, 32, 64]))
NY_R.append(16)



#%% Plotting

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

# Two plots
fig = plt.figure(figsize=(3.5, 3))
axR1 = fig.subplots(1, 1)

# # Two plots
# fig = plt.figure(figsize=(7.75, 3))
# fig.subplots_adjust(hspace=0.5, wspace=0.5)
# axL1, axR1 = fig.subplots(1, 2)

# if len(E_2_L) == 1:
#     cycler = plt.cycler(color=[black], marker=['d'])
# elif len(E_2_L) < 4: # 2 and 3
#     cycler = plt.cycler(color=[blue, red, black], marker=['o', 's', 'd'])
# else: # 4 or more
#     cycler = plt.cycler(color=[blue, red, orange, black, green] + colors[4:],
#         marker=['o', 's', '^', 'd', 'x', '+', 'v', '<', '>', '*', 'p'])

# axL1.set_prop_cycle(cycler)

# N_L = []
# inds_L = []
# for i, error in enumerate(E_2_L):
#     N_L.append(np.log2(NY_L[i]*NX_L[i]**2).astype('int'))
#     inds_L.append(N_L[i] >= 2)
#     axL1.semilogy(N_L[i][inds_L[i]], error[inds_L[i]]/(2*np.pi), label=labels_L[i],
#                   linewidth=solid_linewidth)
# # axL1.minorticks_off()
# Nmin = min([min(N_L[i]) for i in range(len(N_L))])
# Nmax = max([max(N_L[i]) for i in range(len(N_L))])
# Nstep = 2
# axL1.set_xticks(np.linspace(Nmin, Nmax, (Nmax - Nmin)//Nstep + 1))
# axL1.set_title(r'Uniform Grid')
# axL1.set_xlabel(r'$\log_2(N_xN_y)$')
# axL1.set_ylabel(r'$|E_2|$', rotation=0, labelpad=10)
# axL1.legend(loc='lower left')
# xlim = axL1.get_xlim()

if len(E_2_R) == 1:
    cycler = plt.cycler(color=[black], marker=['d'])
elif len(E_2_R) < 4: # 2 and 3
    cycler = plt.cycler(color=[blue, red, black], marker=['o', 's', 'd'])
else: # 4 or more
    cycler = plt.cycler(color=[blue, red, orange, black, green] + colors[4:],
        marker=['o', 's', '^', 'd', 'x', '+', 'v', '<', '>', '*', 'p'])

axR1.set_prop_cycle(cycler)

N_R = []
inds_R = []
for i, error in enumerate(E_2_R):
    N_R.append(np.log2(NY_R[i]*NX_R[i]**2).astype('int'))
    inds_R.append(N_R[i] >= 2)
    axR1.semilogy(N_R[i][inds_R[i]], error[inds_R[i]]/(2*np.pi), label=labels_R[i],
                  linewidth=solid_linewidth)
Nmin = min([min(N_R[i]) for i in range(len(N_R))])
Nmax = max([max(N_R[i]) for i in range(len(N_R))])
Nstep = 2
axR1.set_xticks(np.linspace(Nmin, Nmax, (Nmax - Nmin)//Nstep + 1))
# axR1.set_xlim(xlim)
# axR1.set_title(r'Perturbed Grid (up to 10\%)')
axR1.set_xlabel(r'$\log_2(N_xN_y)$')
axR1.set_ylabel(r'$|E_2|$', rotation=0, labelpad=10)
axR1.legend(loc='lower left')

# ordb = 0
# ordt = 3
# ordstep = 0.5
# axR1.set_ylim(ordb, ordt)
# axR1.set_yticks(np.linspace(ordb, ordt, int((ordt - ordb)/ordstep) + 1))
# lines, labels = axR1.get_legend_handles_labels()
# axR1.legend(lines[1:], labels[1:], loc='lower right')

fig.savefig("AD_conv.pdf", bbox_inches = 'tight', pad_inches = 0)