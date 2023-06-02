# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:55:08 2020

@author: Samuel A. Maloney
"""

import numpy as np
import matplotlib.pyplot as plt

from kernels import *

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
                                r'\usepackage{derivative}'
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

# # Nord
# blue = '#5e81ac'
# orange = '#d08770'
# green = '#a3be8c'
# red = '#bf616a'
# purple = '#b48ead'
# yellow = '#ebcb8b'
# black = '#000000'

# Matplotlib
blue = '#1f77b4' #
orange = '#ff7f0e' #
green = '#2ca02c'
red = '#d62728' #
purple = '#9467bd'
brown = '#8c564b'
pink = '#e377c2'
grey = '#7f7f7f'
cyan = '#17becf'
black = '#000000' #
cycler = plt.cycler(color=[blue, red, orange, black, purple, green, cyan])

fig = plt.figure(figsize=(7.75, 6))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
axes = fig.subplot_mosaic('AA;BC')
ax0 = axes['A']
ax1 = axes['B']
ax2 = axes['C']
for key, ax in axes.items():
    ax.set_prop_cycle(cycler)

xlim = (0,1)
x = np.linspace(xlim[0],xlim[1],100*(xlim[1]-xlim[0]) + 1)

# GenericSpline order
n = 5

##### Values #####
# ax0.plot(x, QuadraticSpline()(np.abs(x)), label='quadratic')
# ax0.plot(x, SimpleCubicSpline()(np.abs(x)), label='simpleCubic')
ax0.plot(x, CubicSpline()(np.abs(x)), label='cubic')
ax0.plot(x, QuarticSpline()(np.abs(x)), label='quartic')
# ax0.plot(x, SimpleQuinticSpline()(np.abs(x)), label='simpleQuintic')
ax0.plot(x, QuinticSpline()(np.abs(x)), label='quintic')
ax0.plot(x, GenericSpline(n=n)(np.abs(x)), label=f'\[(1-r^2)^{n}\]', zorder=0)
# ax0.plot(x, SepticSpline()(np.abs(x)), label='septic')
# ax0.plot(x, Gaussian()(np.abs(x)), label='Gaussian')
ax0.plot(x, Bump()(np.abs(x)), label='bump')
ax0.set_ylabel(r'\[w(r)\]', rotation=0, verticalalignment='center', horizontalalignment='right')
ax0.legend(framealpha=1)
ylim = ax0.get_ylim()
ax0.set_ylim((0, ylim[1]))

#### 1st Derivatives #####
# ax1.plot(x, QuadraticSpline().dw(np.abs(x))[1], label='quadratic')
# ax1.plot(x, SimpleCubicSpline().dw(np.abs(x))[1], label='simpleCubic')
ax1.plot(x, CubicSpline().dw(np.abs(x))[1], label='cubic')
ax1.plot(x, QuarticSpline().dw(np.abs(x))[1], label='quartic')
# ax1.plot(x, SimpleQuinticSpline().dw(np.abs(x))[1], label='simpleQuintic')
ax1.plot(x, QuinticSpline().dw(np.abs(x))[1], label='quintic')
ax1.plot(x, GenericSpline(n=n).dw(np.abs(x))[1], label=f'\[(1-r^2)^{n}\]', zorder=0)
# ax1.plot(x, SepticSpline().dw(np.abs(x))[1], label='septic')
# ax1.plot(x, Gaussian().dw(np.abs(x))[1], label='Gaussian')
ax1.plot(x, Bump().dw(np.abs(x))[1], label='bump')
ax1.set_ylabel(r'\[\odv{w}{r}\]', rotation=0, verticalalignment='center', horizontalalignment='right')

##### 2nd Derivatives #####
# ax2.plot(x, QuadraticSpline().d2w(np.abs(x))[2], label='quadratic')
# ax2.plot(x, SimpleCubicSpline().d2w(np.abs(x))[2], label='simpleCubic')
ax2.plot(x, CubicSpline().d2w(np.abs(x))[2], label='cubic')
ax2.plot(x, QuarticSpline().d2w(np.abs(x))[2], label='quartic')
# ax2.plot(x, SimpleQuinticSpline().d2w(np.abs(x))[2], label='simpleQuintic')
ax2.plot(x, QuinticSpline().d2w(np.abs(x))[2], label='quintic')
ax2.plot(x, GenericSpline(n=n).d2w(np.abs(x))[2], label=f'\[(1-r^2)^{n}\]', zorder=0)
# ax2.plot(x, SepticSpline().d2w(np.abs(x))[2], label='septic')
# ax2.plot(x, Gaussian().d2w(np.abs(x))[2], label='Gaussian')
ax2.plot(x, Bump().d2w(np.abs(x))[2], label='bump')
ax2.set_ylabel(r'\[\frac{\mathrm{d}^2w}{\mathrm{d}r^2}\]', rotation=0, verticalalignment='center', horizontalalignment='right')


for key, ax in axes.items():
    ax.set_xlim(xlim)
    ax.set_xlabel(r'$r$')

bounds1 = ax1.get_position().bounds
bounds0 = ax0.get_position().bounds
shift = 0.5*(bounds0[2] - bounds1[2])
new_bounds = [bounds0[0] + shift, bounds0[1], bounds1[2], bounds0[3]]
ax0.set_position(new_bounds)

# fig.savefig(f"weight_functions_plot.pdf", bbox_inches = 'tight', pad_inches = 0)
