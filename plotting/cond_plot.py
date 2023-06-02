#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:11:57 2023

@author: Samuel A. Maloney
"""

import numpy as np
import matplotlib.pyplot as plt


conds = []
E_2 = []
labels = []

# For all of the below, unless otherwise noted
# xmax = 1.0
# basis = quadratic, kernel = quintic
# boundary = PeriodicBoundary(support = [9.5 9.5])
# NQX = 1, NQY = 1*NY, massLumping = False
# Qord = 4, quadType = uniform
# VCI: VC2-C (whole domain) using ssqr.min2norm

# SinusoidalMapping(0.2, -0.25, 1.0)
# px = py = 0.1, seed = 42
conds.append(np.array([7.36323072e+02, 9.88467785e+02, 1.22003349e+03, 5.58882046e+01,
       1.05789175e+02, 1.07882122e+02, 8.10047552e+01, 5.39887045e+01,
       4.10223844e+01, 3.52932208e+01, 3.13883335e+01, 2.94860408e+01,
       3.03792901e+01, 3.20148304e+01, 8.38983982e+02, 1.22044072e+02,
       1.02991801e+02, 1.04739703e+02, 1.11751331e+02, 1.19987735e+02,
       1.30325354e+02, 1.54940924e+02, 2.14252485e+02, 4.00106416e+02,
       2.44031942e+04, 3.50288404e+02, 1.77619784e+02, 1.22862727e+02,
       9.63196184e+01, 7.97713845e+01, 8.90072523e+01, 1.27257563e+02,
       1.90637057e+02, 3.26892275e+02, 8.74523485e+02, 1.87821912e+03,
       5.14102784e+02, 3.26548222e+02, 2.58424607e+02, 2.28348699e+02,
       2.16121602e+02, 2.14442408e+02, 2.20091940e+02, 2.31440114e+02,
       2.47541281e+02, 2.67759073e+02, 2.91632853e+02, 3.18914893e+02,
       3.49757133e+02, 3.84956633e+02, 4.25957960e+02, 4.74056715e+02,
       5.28350664e+02, 5.82880784e+02, 6.25163822e+02, 6.40070666e+02,
       6.19980990e+02, 5.72040168e+02, 5.12861349e+02, 4.57210691e+02,
       4.12369370e+02, 3.79263101e+02, 3.55771484e+02, 3.39216934e+02,
       3.27425529e+02, 3.18890764e+02, 3.12626383e+02, 3.07990465e+02,
       3.04555734e+02, 3.02026733e+02, 3.00189786e+02, 2.98883451e+02,
       2.97981239e+02, 2.97381497e+02, 2.97001393e+02, 2.96773179e+02,
       2.96641714e+02, 2.96562630e+02, 2.96500850e+02, 2.96429302e+02,
       2.96327740e+02]))
E_2.append(np.array([1.71361526e+00, 4.04267668e+01, 1.19673136e+01, 8.42192358e-02,
       1.43360503e-01, 1.58448794e-01, 1.38752572e-01, 1.08330133e-01,
       7.88935053e-02, 5.69210237e-02, 5.49630159e-02, 7.91181993e-02,
       1.18194235e-01, 1.69376373e-01, 1.27581530e+00, 2.37296650e-01,
       2.23036933e-01, 2.17940196e-01, 2.07362344e-01, 2.03107542e-01,
       2.42214529e-01, 3.25451750e-01, 4.84341772e-01, 9.52197369e-01,
       6.05274502e+01, 9.27538208e-01, 5.36955609e-01, 4.32042205e-01,
       3.63090865e-01, 2.96027019e-01, 2.47660129e-01, 2.36615377e-01,
       2.73715020e-01, 3.86937323e-01, 8.94897307e-01, 1.94257301e+00,
       6.12919295e-01, 4.44511692e-01, 3.79368229e-01, 3.45395630e-01,
       3.29651888e-01, 3.29368531e-01, 3.43105657e-01, 3.68763078e-01,
       4.03772090e-01, 4.45816644e-01, 4.93337930e-01, 5.45730627e-01,
       6.03345175e-01, 6.67337166e-01, 7.39219347e-01, 8.19707989e-01,
       9.06275528e-01, 9.89395303e-01, 1.05003819e+00, 1.06473644e+00,
       1.02088132e+00, 9.28797355e-01, 8.14611139e-01, 7.02862907e-01,
       6.07305802e-01, 5.32035838e-01, 4.75835734e-01, 4.35492308e-01,
       4.07515166e-01, 3.88826144e-01, 3.76942231e-01, 3.69947796e-01,
       3.66398482e-01, 3.65218082e-01, 3.65610330e-01, 3.66989939e-01,
       3.68930645e-01, 3.71126404e-01, 3.73362292e-01, 3.75492416e-01,
       3.77422961e-01, 3.79099051e-01, 3.80494508e-01, 3.81603862e-01,
       3.82436127e-01]))
labels.append(r'sin. \percent{10}')

# SinusoidalMapping(0.2, -0.25, 1.0)
# px = py = 0, seed = 42
conds.append(np.array([2.38358011e+01, 1.70983198e+01, 2.42007906e+01, 3.21668050e+01,
       9.93448655e+01, 1.10735394e+02, 8.58170907e+01, 5.77772117e+01,
       4.15234777e+01, 3.51640394e+01, 3.08911100e+01, 2.84763992e+01,
       2.84354509e+01, 4.80099938e+01, 1.84866703e+02, 8.74020113e+01,
       7.67866569e+01, 7.36818556e+01, 7.21118025e+01, 7.13974195e+01,
       7.26543426e+01, 8.19014133e+01, 1.01193378e+02, 1.38702388e+02,
       2.35082596e+02, 8.81124546e+02, 4.74678607e+02, 1.86127614e+02,
       1.18335707e+02, 1.01156265e+02, 1.29503054e+02, 1.72089493e+02,
       2.56640875e+02, 5.21000160e+02, 2.10008167e+04, 5.27742245e+02,
       2.82779460e+02, 2.03009607e+02, 1.65229806e+02, 1.44342450e+02,
       1.31932196e+02, 1.24392605e+02, 1.19932113e+02, 1.17569914e+02,
       1.16730636e+02, 1.17056034e+02, 1.18305111e+02, 1.20292632e+02,
       1.22845980e+02, 1.25779540e+02, 1.28907511e+02, 1.32125663e+02,
       1.35524307e+02, 1.39353190e+02, 1.43777056e+02, 1.48735335e+02,
       1.54044211e+02, 1.59520386e+02, 1.65021850e+02, 1.70444934e+02,
       1.75713605e+02, 1.80771355e+02, 1.85576264e+02, 1.90098064e+02,
       1.94316213e+02, 1.98218477e+02, 2.01799707e+02, 2.05060735e+02,
       2.08007327e+02, 2.10649203e+02, 2.12999135e+02, 2.15072136e+02,
       2.16884742e+02, 2.18454400e+02, 2.19798958e+02, 2.20936251e+02,
       2.21883773e+02, 2.22658427e+02, 2.23276352e+02, 2.23752795e+02,
       2.24102034e+02]))
E_2.append(np.array([8.21625593e-01, 1.13237954e+00, 1.60173994e+01, 1.46259598e-01,
       1.28927541e-01, 1.57376968e-01, 1.43167762e-01, 1.15459043e-01,
       8.18030194e-02, 4.78411839e-02, 3.70919558e-02, 7.55601607e-02,
       1.29429287e-01, 1.86935109e-01, 2.89001692e-01, 2.39903865e-01,
       2.23605538e-01, 2.07753517e-01, 1.96493974e-01, 1.92990447e-01,
       1.98807989e-01, 2.15801913e-01, 2.49319989e-01, 3.16699453e-01,
       4.91260806e-01, 1.69380513e+00, 8.67645583e-01, 3.49341673e-01,
       2.51405066e-01, 2.27599849e-01, 2.25469559e-01, 2.30159743e-01,
       2.39496194e-01, 2.76428420e-01, 6.95118477e+00, 3.19929215e-01,
       2.78343447e-01, 2.70382791e-01, 2.68492105e-01, 2.68341212e-01,
       2.68709473e-01, 2.69139596e-01, 2.69431913e-01, 2.69492446e-01,
       2.69275480e-01, 2.68757448e-01, 2.67922663e-01, 2.66754308e-01,
       2.65229669e-01, 2.63321157e-01, 2.61006029e-01, 2.58284736e-01,
       2.55198877e-01, 2.51832487e-01, 2.48289266e-01, 2.44659875e-01,
       2.41001609e-01, 2.37338193e-01, 2.33671292e-01, 2.29992889e-01,
       2.26293476e-01, 2.22565939e-01, 2.18806755e-01, 2.15015921e-01,
       2.11196469e-01, 2.07353943e-01, 2.03495996e-01, 1.99632096e-01,
       1.95773337e-01, 1.91932314e-01, 1.88123033e-01, 1.84360837e-01,
       1.80662327e-01, 1.77045265e-01, 1.73528453e-01, 1.70131581e-01,
       1.66875042e-01, 1.63779693e-01, 1.60866591e-01, 1.58156668e-01,
       1.55670372e-01]))
labels.append(r'sin. uniform')

# StraightMapping()
# px = py = 0.1, seed = 42
conds.append(np.array([9.10620342e+02, 2.09133751e+02, 1.28898901e+01, 2.13848402e+01,
       2.83864097e+01, 3.64254250e+01, 4.11868336e+01, 4.24650781e+01,
       4.07352006e+01, 3.75554843e+01, 3.37627045e+01, 3.02206640e+01,
       2.67547905e+01, 2.35961456e+01, 2.11629032e+01, 1.95253272e+01,
       2.28964802e+01, 3.70407721e+01, 6.98080342e+01, 1.59450778e+02,
       4.75632745e+02, 2.53949217e+03, 6.53149121e+03, 1.64516732e+03,
       7.43283774e+02, 4.84604265e+02, 3.82014033e+02, 3.47363325e+02,
       3.53471731e+02, 3.99799187e+02, 4.64620878e+02, 5.79437941e+02,
       7.65962167e+02, 8.50151177e+02, 8.41533167e+02, 8.21252400e+02,
       8.02986631e+02, 8.43836867e+02, 9.67028811e+02, 1.18725068e+03,
       1.54025250e+03, 2.10177075e+03, 3.00351398e+03, 4.36584730e+03,
       5.79831116e+03, 6.01895564e+03, 5.08857683e+03, 4.61658299e+03,
       5.42502589e+03, 7.24032704e+03, 9.35992548e+03, 1.07482778e+04,
       1.08461357e+04, 1.00239958e+04, 8.90220949e+03, 7.83869021e+03,
       6.94873129e+03, 6.23834748e+03, 5.68177992e+03, 5.25183153e+03,
       4.92915901e+03, 4.70387072e+03, 4.57466472e+03, 4.54766734e+03,
       4.63642557e+03, 4.86523380e+03, 5.28122066e+03, 5.99130308e+03,
       7.28161612e+03, 1.01171917e+04, 2.06492800e+04, 3.89174590e+04,
       6.62403049e+03, 3.54018973e+03, 4.00199162e+03, 3.77860767e+04,
       4.20592589e+03, 2.27037453e+03, 1.69331516e+03, 1.36954925e+03,
       1.13418638e+03]))
E_2.append(np.array([3.41223605e+00, 1.19034702e+00, 3.04328314e-01, 1.83837071e-01,
       9.73092921e-02, 4.34771853e-02, 2.31627132e-02, 1.84024105e-02,
       1.85995465e-02, 2.26776688e-02, 2.99126038e-02, 3.89105807e-02,
       4.87636332e-02, 5.92412383e-02, 7.27520882e-02, 8.99111477e-02,
       1.09329137e-01, 1.29001435e-01, 1.49034937e-01, 1.77577663e-01,
       1.96404927e-01, 2.12212224e-01, 2.49254059e-01, 3.46163356e-01,
       4.60617502e-01, 4.84395836e-01, 4.46842621e-01, 3.84029475e-01,
       3.88286563e-01, 3.52425686e-01, 3.45569376e-01, 3.41635269e-01,
       3.53571251e-01, 3.75678982e-01, 3.91404299e-01, 3.95471534e-01,
       3.97417548e-01, 4.00117153e-01, 4.02991616e-01, 4.05072415e-01,
       4.05731372e-01, 4.05182624e-01, 4.05884941e-01, 4.15849033e-01,
       4.43769583e-01, 4.39078872e-01, 4.05577903e-01, 5.17929019e-01,
       5.96028695e-01, 6.28815128e-01, 6.53606543e-01, 6.82593927e-01,
       7.13402085e-01, 7.39461093e-01, 7.57700946e-01, 7.68594368e-01,
       7.73420701e-01, 7.73105549e-01, 7.68262941e-01, 7.59578111e-01,
       7.48259288e-01, 7.36566556e-01, 7.28447179e-01, 7.30087970e-01,
       7.49565724e-01, 7.93680525e-01, 8.59170754e-01, 9.15639441e-01,
       8.75152868e-01, 5.48133739e-01, 2.91451602e+00, 1.97391908e+01,
       8.70534265e+00, 8.89932378e+00, 1.33436347e+01, 1.23893225e+02,
       1.11454975e+01, 4.50152032e+00, 2.58863723e+00, 1.80007943e+00,
       1.48706168e+00]))
labels.append(r'str. \percent{10}')

# StraightMapping()
# px = py = 0, seed = 42
conds.append(np.array([8.41663398e+00, 8.51286725e+00, 8.68250523e+00, 1.26022613e+01,
       1.99870638e+01, 2.82500137e+01, 3.47041446e+01, 3.78851477e+01,
       3.70872963e+01, 3.37618475e+01, 2.96560535e+01, 2.57403020e+01,
       2.25560570e+01, 2.01846220e+01, 1.83815963e+01, 1.70322330e+01,
       2.23262482e+01, 3.59403301e+01, 6.76920748e+01, 1.54322980e+02,
       4.71108087e+02, 2.45652169e+03, 5.08703947e+03, 1.46253484e+03,
       7.24946458e+02, 4.94149016e+02, 4.00386351e+02, 3.62331629e+02,
       3.56292788e+02, 3.77849426e+02, 4.36289762e+02, 5.58827784e+02,
       7.72873848e+02, 9.84682472e+02, 1.01212267e+03, 9.71499940e+02,
       1.01300252e+03, 1.16414720e+03, 1.44240096e+03, 1.90486678e+03,
       2.68670313e+03, 4.10864382e+03, 7.04985715e+03, 1.45964229e+04,
       4.13916706e+04, 8.70311726e+04, 3.15885945e+04, 1.33282250e+04,
       7.63644117e+03, 5.06037594e+03, 3.63053920e+03, 2.74876899e+03,
       2.16774194e+03, 1.76564493e+03, 1.47628164e+03, 1.26123918e+03,
       1.09707251e+03, 9.68862317e+02, 8.66770707e+02, 7.84107460e+02,
       7.16198658e+02, 6.59700623e+02, 6.12170514e+02, 5.71789903e+02,
       5.37182286e+02, 5.07289845e+02, 4.81288509e+02, 4.58528308e+02,
       4.38490776e+02, 4.20758060e+02, 4.04990197e+02, 3.90908175e+02,
       3.78281160e+02, 3.66916761e+02, 3.56653513e+02, 3.47355046e+02,
       3.38905496e+02, 3.31205878e+02, 3.24171189e+02, 3.17728086e+02,
       3.11812994e+02]))
E_2.append(np.array([3.70256899e-01, 3.47481974e-01, 3.34364683e-01, 2.43082363e-01,
       1.40242665e-01, 7.26236165e-02, 3.71993766e-02, 2.30867188e-02,
       2.17187435e-02, 2.71385457e-02, 3.56924069e-02, 4.57073764e-02,
       5.67282689e-02, 6.86344616e-02, 8.12579872e-02, 9.43603322e-02,
       1.07660053e-01, 1.20847653e-01, 1.33649617e-01, 1.45865361e-01,
       1.57367918e-01, 1.68080235e-01, 1.78123430e-01, 1.87403309e-01,
       1.95897522e-01, 2.03716454e-01, 2.10921106e-01, 2.17584525e-01,
       2.23806752e-01, 2.29747570e-01, 2.35703788e-01, 2.42240336e-01,
       2.50017775e-01, 2.57985616e-01, 2.62903707e-01, 2.64507633e-01,
       2.65198941e-01, 2.66084259e-01, 2.67227596e-01, 2.68495451e-01,
       2.69775794e-01, 2.70995942e-01, 2.72109497e-01, 2.73089391e-01,
       2.74035135e-01, 2.75098418e-01, 2.76058655e-01, 2.76772399e-01,
       2.77147233e-01, 2.77328595e-01, 2.77363004e-01, 2.77261149e-01,
       2.77022899e-01, 2.76645992e-01, 2.76128221e-01, 2.75467800e-01,
       2.74663339e-01, 2.73713772e-01, 2.72618310e-01, 2.71376416e-01,
       2.69987801e-01, 2.68452432e-01, 2.66770541e-01, 2.64942644e-01,
       2.62969557e-01, 2.60852408e-01, 2.58592653e-01, 2.56192088e-01,
       2.53652855e-01, 2.50977453e-01, 2.48168743e-01, 2.45229947e-01,
       2.42164657e-01, 2.38976828e-01, 2.35670782e-01, 2.32251206e-01,
       2.28723149e-01, 2.25092021e-01, 2.21363590e-01, 2.17543986e-01,
       2.13639695e-01]))
labels.append(r'str. uniform')


start = 1.5
stop = 9.5
step = 0.1
nSamples = int(np.rint((stop - start)/step)) + 1
support_size_array = np.linspace(start, stop, num=nSamples)


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
                                r'\newcommand*{\norm}[1]{\left\lVert#1\right\rVert}'
                                r'\newcommand*{\percent}[1]{\ensuremath{#1\,\%}}'
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

nSeries = len(conds)

if nSeries == 1:
    cycler = plt.cycler(color=[black])
elif nSeries == 2:
    cycler = plt.cycler(color=[blue, red])
elif nSeries == 3:
    cycler = plt.cycler(color=[blue, red, black])
elif nSeries == 4:
    cycler = plt.cycler(color=[blue, red, orange, black])
else:
    pass # should never be more than 4

# clear the current figure, if opened, and set parameters
fig = plt.figure(figsize=(7.75, 3))
axes = fig.subplots(1,2)
fig.subplots_adjust(hspace=0.3, wspace=0.3)

for ax in axes:
    ax.set_prop_cycle(cycler)

for i, cond in enumerate(conds):
    axes[0].plot(support_size_array, cond[:nSamples], label=labels[i])
    axes[0].set_ylabel('Condition Number')
    axes[0].set_ylim((0,1000))

    axes[1].plot(support_size_array, E_2[i][:nSamples])
    axes[1].set_ylabel(r'$\norm{u-u^d}$')
    axes[1].set_ylim((0,1))

axes[0].legend(framealpha=1)

for ax in axes:
    # for size in [3,4,5]:
    #     ax.axvline(size, linestyle=':', color='black')#, linewidth=dashed_linewidth)
    
    # ax.set_xlabel('support size')
    ax.set_xticks(np.arange(np.trunc(start), np.ceil(stop)+1).astype('int'))

fig.supxlabel('Support Size as a Multiple of Uniform Grid Spacing', verticalalignment='top')

# fig.savefig("fcimls_cond_plot.pdf", bbox_inches = 'tight', pad_inches = 0)
