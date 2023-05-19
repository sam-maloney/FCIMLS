# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:46:29 2021

@author: Samuel A. Maloney

"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

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
# NQX = 1, quadType = uniform

# Qord = 3

# # StraightMapping()
# E_2_L.append(np.)
# E_inf_L.append(np.)
# labels_L.append(r'uniform  str')
# NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_L.append(1)

# # StraightMapping()
# E_2_L.append(np.)
# E_inf_L.append(np.)
# labels_L.append(r'\percent{10} pert.  str')
# NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_L.append(1)

# # StraightMapping()
# E_2_L.append(np.)
# E_inf_L.append(np.)
# labels_L.append(r'\percent{50} pert.  str')
# NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_L.append(1)

# SinusoidalMapping(0.2, -0.25, 1.0)
E_2_L.append(np.array([9.36764871e-02, 1.33868722e-02, 1.74381040e-04, 1.29907132e-05,
        1.11008161e-06, 7.58009105e-08, 4.85362063e-09]))
E_inf_L.append(np.array([1.87830658e-01, 3.64948239e-02, 3.15902038e-04, 3.25672004e-05,
        3.09999045e-06, 2.16442766e-07, 1.39023197e-08]))
labels_L.append(r'uniform')
NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_L.append(1)

# SinusoidalMapping(0.2, -0.25, 1.0)
E_2_L.append(np.array([7.47307916e-02, 1.29563336e-02, 1.22376963e-03, 1.95764540e-04,
       2.50340612e-05, 3.25294226e-06, 4.48321879e-07]))
E_inf_L.append(np.array([1.42157826e-01, 3.62033762e-02, 3.61658745e-03, 7.45368712e-04,
       1.16870750e-04, 1.86108782e-05, 2.49211728e-06]))
labels_L.append(r'\percent{10} pert.')
NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_L.append(1)

# SinusoidalMapping(0.2, -0.25, 1.0)
E_2_L.append(np.array([1.37097881e-01, 3.75875416e-02, 5.57553607e-03, 7.42579943e-04,
       9.04138873e-05, 1.06317095e-05, 1.58003755e-06]))
E_inf_L.append(np.array([2.39390305e-01, 1.15165557e-01, 1.71982630e-02, 2.63195434e-03,
       4.27859051e-04, 6.55303608e-05, 1.15043639e-05]))
labels_L.append(r'\percent{50} pert.')
NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_L.append(1)

# Qord = 4
# # SinusoidalMapping(0.2, -0.25, 1.0)
# E_2_L.append(np.array([1.31495937e-01, 3.87986766e-02, 5.55124158e-03, 7.33094490e-04,
#        9.01388754e-05, 1.13076646e-05, 1.59739105e-06]))
# E_inf_L.append(np.array([2.43668714e-01, 1.20885712e-01, 1.67880333e-02, 2.92186535e-03,
#        4.18979311e-04, 7.78691694e-05, 1.16376115e-05]))
# labels_L.append(r'\percent{50} pert.  sin, Q4u')
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
# massLumping = False, quadType = uniform
# VCI: None using None

# Nx = Ny

# # boundary = DirichletBoundary(support = [3. 3.], NDX = 3)
# E_2_R.append(np.array([1.55623497e-01, 2.00444684e-02, 1.07843261e-03, 1.48710801e-04,
#         1.18687318e-05, 9.94947319e-07, 8.61300145e-08]))
# E_inf_R.append(np.array([5.41963727e-01, 1.31031295e-01, 6.18536899e-03, 1.14104369e-03,
#         1.17101126e-04, 1.31643834e-05, 1.72769561e-06]))
# labels_R.append(r'uniform \ratio{1}{1}, $\text{NDX}=3$')
# NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_R.append(1)

# # boundary = DirichletBoundary(support = [3. 3.], NDX=2)
# E_2_R.append(np.array([2.31915184e-01, 2.01695351e-02, 1.37079270e-03, 1.69842804e-04,
#        1.57430093e-05, 1.41632461e-06, 1.25707142e-07]))
# E_inf_R.append(np.array([8.87431012e-01, 1.23553933e-01, 9.34272434e-03, 1.44784772e-03,
#        1.79068982e-04, 3.26113758e-05, 4.70586697e-06]))
# labels_R.append(r'uniform \ratio{1}{1}, $\text{NDX}=2$')
# NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_R.append(1)

### note that support = [3. 3.] fails when nodes perturbed, insufficient cover

# boundary = DirichletBoundary(support = [3.5 3.5], NDX=2)
E_2_R.append(np.array([2.20436448e-01, 9.17532004e-03, 9.29017875e-04, 1.47930838e-04,
       1.47996047e-05, 1.38510694e-06]))
E_inf_R.append(np.array([9.23390306e-01, 3.55543361e-02, 5.88191576e-03, 1.35730901e-03,
       1.70417833e-04, 3.29410441e-05]))
labels_R.append(r'uniform')
NX_R.append(np.array([  4,   8,  16,  32,  64, 128]))
NY_R.append(1)

# boundary = DirichletBoundary(support = [3.5 3.5], NDX=2)
E_2_R.append(np.array([1.96638754e-01, 8.58787575e-03, 2.02797040e-03, 3.21173569e-04,
       4.28788383e-05, 4.81298543e-06]))
E_inf_R.append(np.array([8.74730221e-01, 3.08983671e-02, 1.11169466e-02, 2.38065736e-03,
       3.65347307e-04, 5.43917647e-05]))
labels_R.append(r'\percent{10} pert.')
NX_R.append(np.array([  4,   8,  16,  32,  64, 128]))
NY_R.append(1)

# boundary = DirichletBoundary(support = [3.5 3.5], NDX=2)
E_2_R.append(np.array([2.63138819e-01, 2.42072002e-02, 8.91164560e-03, 1.42974221e-03,
       1.57403761e-04, 2.04670969e-05]))
E_inf_R.append(np.array([1.16741599e+00, 8.73488108e-02, 6.31503600e-02, 6.09721370e-03,
       1.28310033e-03, 1.99328264e-04]))
labels_R.append(r'\percent{50} pert.')
NX_R.append(np.array([  4,   8,  16,  32,  64, 128]))
NY_R.append(1)




##### Begin Plotting Routines #####

solid_linewidth = 1.25
dashed_linewidth = 1.0

plt.rc('markers', fillstyle='full')
plt.rc('lines', markersize=5.0, linewidth=solid_linewidth)
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
                                r'\newcommand*{\percent}[1]{\ensuremath{#1\,\%}}'
                                r'\newcommand*{\norm}[1]{\left\lVert#1\right\rVert}'
                                # r'\newcommand*{\normalise}[1]{\frac{#1}{\max\left\lvert#1\right\rvert}}'
                                r'\newcommand*{\normalise}[1]{\frac{#1}{\norm{#1}}_\infty}'
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

if len(E_2_L) == 1:
    cycler = plt.cycler(color=[black], marker=['d'])
elif len(E_2_L) == 2:
    cycler = plt.cycler(color=[blue, red], marker=['o', 's'])
else:
    cycler = plt.cycler(color=[black, blue, red, orange, green] + colors[4:], 
        marker=['d', 'o', 's', '^', 'x', '+', 'v', '<', '>', '*', 'p'])

# fig = plt.figure(figsize=(7.75, 3))
# fig.subplots_adjust(hspace=0.5, wspace=0.5)

##### All the plots! #####
fig = plt.figure(figsize=(7.75, 9))
axes = fig.subplots(3, 2)
# axL1, axR1 = fig.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1.1]})
axL1 = axes[0,0]
axR1 = axes[0,1]

fig.subplots_adjust(hspace=0.5, wspace=0.6)

# right_title_kwargs={'rotation' : 90, 'labelpad' : 4.0}
right_title_kwargs={'rotation' : 0, 'labelpad' : 0,
    'verticalalignment': 'center', 'horizontalalignment' : 'left'}

solutioncmap = 'viridis'
# solutioncmap = 'inferno'

f = sinXsinY()
nx = ny = 128
X, Y = np.meshgrid((1/nx)*np.arange(nx+1), (1/ny)*np.arange(ny+1))
U = f.solution(np.hstack((X.reshape(-1,1), Y.reshape(-1,1))))*(1/f.umax)
field = axes[1,0].tripcolor(X.ravel(), Y.ravel(), U, shading='gouraud',
                            cmap=solutioncmap, vmin=-1, vmax=1)
cbar = fig.colorbar(field, ax=axes[1,0])
cbar.set_ticks(np.linspace(-1,1,5))
cbar.set_label(r'$f(x,y)$', **right_title_kwargs)
# x = np.linspace(0, sim.nodeX[-1], 100)
# axes[1,0].plot(x, [sim.mapping(np.array([[0, float(yi)]]), i) for i in x], 'k')

f = QuadraticTestProblem()
U = f.solution(np.hstack((X.reshape(-1,1), Y.reshape(-1,1))))*(1/f.umax)
field = axes[1,1].tripcolor(X.ravel(), Y.ravel(), U, shading='gouraud',
                            cmap=solutioncmap, vmin=-1, vmax=1)
cbar = fig.colorbar(field, ax=axes[1,1])
cbar.set_ticks(np.linspace(-1,1,5))
cbar.set_label(r'$f(x,y)$', **right_title_kwargs)

errorcmap = 'RdBu'
# errorcmap = 'seismic'

fileE = open('quad_proj_error.dat', 'rb')
E = pickle.load(fileE)
fileE.close()
nx = ny = int(np.sqrt(E.size) - 1)
X, Y = np.meshgrid((1/nx)*np.arange(nx+1), (1/ny)*np.arange(ny+1))
X = X.T.ravel()
Y = Y.T.ravel()
maxAbsE = np.max(np.abs(E))
field = axes[2,1].tripcolor(X.ravel(), Y.ravel(), E/maxAbsE, shading='gouraud',
                            cmap=errorcmap, vmin=-1, vmax=1)
cbar = fig.colorbar(field, ax=axes[2,1])
cbar.set_label(r'\[\normalise{f-u}\]', **right_title_kwargs)

fileE = open('sin_proj_error.dat', 'rb')
E = pickle.load(fileE)
fileE.close()
# nx = ny = int(np.sqrt(E.size) - 1)
# X, Y = np.meshgrid((1/nx)*np.arange(nx+1), (1/ny)*np.arange(ny+1))
# X = X.T.ravel()
# Y = Y.T.ravel()
maxAbsE = np.max(np.abs(E))
field = axes[2,0].tripcolor(X.ravel(), Y.ravel(), E/maxAbsE, shading='gouraud',
                            cmap=errorcmap, vmin=-1, vmax=1)
cbar = fig.colorbar(field, ax=axes[2,0])
cbar.set_label(r'\[\normalise{f-u}\]', **right_title_kwargs)

fig.text(0.5,0.613,'Normalised Exact Solution Functions', fontsize='large',
         horizontalalignment='center', verticalalignment='center')
fig.text(0.5,0.323,r'Normalised Uniform Errors for $N_x=N_y=32$', fontsize='large',
         horizontalalignment='center', verticalalignment='center')

for ax in axes[1:].ravel():
    ax.margins(0,0)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$', verticalalignment='center', rotation=0, labelpad=13)
    ax.set_aspect(1)
    

left_title_kwargs={'rotation' : 90, 'labelpad' : 4.0}
# left_title_kwargs={'rotation' : 0, 'labelpad' : 4.0,
#     'verticalalignment': 'center', 'horizontalalignment' : 'right'}

# axL1, axR1 = fig.subplots(1, 2)
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
axL1.set_ylabel(r'$\norm{f-u}$', **left_title_kwargs)
axL2.set_ylabel(r'Intra-step Order of Convergence')
leg = axL1.legend(loc='lower left', framealpha=1)
leg.remove()
Nmin = min([min(N[i]) for i in range(len(N))])
Nmax = max([max(N[i]) for i in range(len(N))])
Nstep = 2
axL1.set_xticks(np.linspace(Nmin, Nmax, (Nmax - Nmin)//Nstep + 1))
ordb = 1
ordt = 5
ordstep = 1
axL2.set_ylim(ordb, ordt)
axL2.set_yticks(np.linspace(ordb, ordt, int((ordt - ordb)/ordstep) + 1))
axL2.add_artist(leg)

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
axR1.set_ylabel(r'$\norm{f-u}$', **left_title_kwargs)
axR2.set_ylabel(r'Intra-step Order of Convergence')
leg = axR1.legend(loc='upper right', framealpha=1)
leg.remove()
Nmin = min([min(N[i]) for i in range(len(N))])
Nmax = max([max(N[i]) for i in range(len(N))])
Nstep = 2
axR1.set_xticks(np.linspace(Nmin, Nmax, (Nmax - Nmin)//Nstep + 1))
axR2.set_ylim(ordb, ordt)
axR2.set_yticks(np.linspace(ordb, ordt, int((ordt - ordb)/ordstep) + 1))
axR1.set_ylim(top=2)
axR2.add_artist(leg)

# fig.savefig('fcimls_projection_conv.pdf', bbox_inches='tight', pad_inches=0)
