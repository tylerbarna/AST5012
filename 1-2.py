# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 09:24:27 2022

@author: conta
"""
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# SciPy bits we use for analysis

from scipy.signal import argrelmin, argrelmax
from scipy import stats

import warnings
warnings.filterwarnings('ignore',category=UserWarning, append=True)

### plotting format
# graphic aspect ratio = width/height
aspect = 4.0/3.0

# Text width in inches - don't change, this is defined by the print layout
textWidth = 6.0 # inches

# output format and resolution
figFmt = 'png'
dpi = 600

# Graphic dimensions 

plotWidth = dpi*textWidth
plotHeight = plotWidth/aspect
axisFontSize = 14
labelFontSize = 10
lwidth = 0.5
axisPad = 5
wInches = textWidth 
hInches = wInches/aspect

# LaTeX is used throughout for markup of symbols, Times-Roman serif font

#plt.rc('text', usetex=True)
plt.rc('font', **{'family':'serif','serif':['Times-Roman'],'weight':'bold','size':'16'})

# Font and line weight defaults for axes

matplotlib.rc('axes',linewidth=lwidth)
matplotlib.rcParams.update({'font.size':axisFontSize})

# axis and label padding

plt.rcParams['xtick.major.pad'] = f'{axisPad}'
plt.rcParams['ytick.major.pad'] = f'{axisPad}'
plt.rcParams['axes.labelpad'] = f'{axisPad}'
#%% Original parameter set from file
xe = 0.001 #fractional ionization of 1e-3
minT = 10.0 #min temp K
maxT = 20000. #max temp K
gain = 20.0 #2 * 10^-26 erg/s (new units?) -> nope, this is because of the weird
#way they implemented the scaling of 10^-27 erg cm^3 / s

# Boltzmann Constant (CODATA 2018)

k = 1.380649e-16 # erg K^-1

minLogT = math.log10(minT)
maxLogT = math.log10(maxT)

logT = np.linspace(minLogT,maxLogT,num=1001)
T = 10.0**logT

xfac = xe/0.001 #this is the factor of x/10^-3 used in the equations
TH = 118000.0 # hydrogen excitation temperature in K
TC = 91.2     # carbon excitation temperature in K
TO = 228.0    # oxygen excitation temperature in K

# Lyman-alpha cooling

coolLya = 6.0e5*(xfac/np.sqrt(T/1.0e4))*np.exp(-TH/T)

# Carbon cooling

coolC = 3.1*(xfac/np.sqrt(T/100.0))*np.exp(-TC/T) + 5.2*((T/100.0)**0.13)*np.exp(-TC/T)

# Oxygen cooling

coolO = 4.1*((T/100.0)**0.42)*np.exp(-TO/T)

# Total cooling

coolTot = (coolLya + coolC + coolO)

# equilibrium density

neq = gain/coolTot

# pressure

P = neq*k*T

#%% 1.10 FGH plot
##### plotting figure 1.10 -> uses C and O abundances in table 1.2, 0.68 and 0.24 %
plotFile = f'Fig1_10.{figFmt}'

fig, ax = plt.subplots()

fig.set_dpi(dpi)
fig.set_size_inches(wInches,hInches,forward=True)

ax.tick_params('both',length=6,width=lwidth,which='major',
               direction='in',top='on',right='on')
ax.tick_params('both',length=3,width=lwidth,which='minor',
               direction='in',top='on',right='on')

# Limits

minCool = 1.0e-30 # erg cm^3 s^-1
maxCool = 1.0e-24
# Labels

xLabel = r'Temperature [K]'
yLabel = r'$\Lambda$ [erg cm$^3$ s$^{-1}$]'

plt.xlim(minT,maxT)
ax.set_xscale('log')
ax.set_xticks([10,100,1000,1.0e4])
ax.set_xticklabels(['10','100','1000','10$^{4}$'])
plt.xlabel(xLabel)

plt.ylim(minCool,maxCool)
ax.set_yscale('log')
ax.set_yticks([1.0E-30,1.0E-29,1.0E-28,1.0E-27,1.0e-26,1.0e-25,1.0e-24])
ax.set_yticklabels(['$10^{-30}$','10$^{-29}$','10$^{-28}$',
                    '10$^{-27}$','10$^{-26}$','10$^{-25}$','10$^{-24}$'])
plt.ylabel(yLabel)

# Plot the total and individual cooling functions

plt.plot(T,1.0e-27*coolTot,'-',color='black',lw=2,zorder=10)
plt.plot(T,1.0e-27*coolLya,'--',color='black',lw=1,zorder=10)
plt.plot(T,1.0e-27*coolC,':',color='black',lw=1,zorder=10)
plt.plot(T,1.0e-27*coolO,'-.',color='black',lw=1,zorder=10)

# label components

lfs = np.rint(0.8*axisFontSize)
plt.text(1000.0,1.7e-26,'Total',fontsize=lfs,rotation=10.0,ha='center',va='bottom')
plt.text(80.0,1.0e-28,r'[O I] $\lambda63\mu m$',fontsize=lfs) #r'$[\textsc{O\,i}]\,\lambda$63$\mu m$'
plt.text(3000.0,3.5e-27,r'[C II] $\lambda 158 \mu m$',
         fontsize=lfs,rotation=3.0,ha='center') #r'$[\textsc{C\,ii}]\,\lambda$158$\mu m$'
plt.text(5400.0,1.0e-28,r'Ly$\alpha$',fontsize=lfs,ha='center')
plt.title("Book plot")
# make the figure

plt.plot()
plt.savefig(plotFile,bbox_inches='tight',facecolor='white')

#%% 1.10 replotting with 10% abundances of C and O as local ISM
#it seems like the only thing that actually depends on the abundance is an unexplained
#coefficient so i'm literally just taking 10% off the top of C and O

coolC_10 = 0.1 * coolC
coolO_10 = 0.1 * coolO
coolTot_10 = (coolLya + coolC_10 + coolO_10)

plotFile = f'Fig1_10_ten_percent.{figFmt}'

fig, ax = plt.subplots()

fig.set_dpi(dpi)
fig.set_size_inches(wInches,hInches,forward=True)

ax.tick_params('both',length=6,width=lwidth,which='major',
               direction='in',top='on',right='on')
ax.tick_params('both',length=3,width=lwidth,which='minor',
               direction='in',top='on',right='on')

# Limits

minCool = 1.0e-30 # erg cm^3 s^-1
maxCool = 1.0e-24
# Labels

xLabel = r'Temperature [K]'
yLabel = r'$\Lambda$ [erg cm$^3$ s$^{-1}$]'

plt.xlim(minT,maxT)
ax.set_xscale('log')
ax.set_xticks([10,100,1000,1.0e4])
ax.set_xticklabels(['10','100','1000','10$^{4}$'])
plt.xlabel(xLabel)

plt.ylim(minCool,maxCool)
ax.set_yscale('log')
ax.set_yticks([1.0E-30,1.0E-29,1.0E-28,1.0E-27,1.0e-26,1.0e-25,1.0e-24])
ax.set_yticklabels(['$10^{-30}$','10$^{-29}$','10$^{-28}$',
                    '10$^{-27}$','10$^{-26}$','10$^{-25}$','10$^{-24}$'])
plt.ylabel(yLabel)

# Plot the total and individual cooling functions

plt.plot(T,1.0e-27*coolTot_10,'-',color='black',lw=2,zorder=10, 
         label = 'Total')
plt.plot(T,1.0e-27*coolLya,'--',color='black',lw=1,zorder=10, 
         label = r'[O I] $\lambda63\mu m$')
plt.plot(T,1.0e-27*coolC_10,':',color='black',lw=1,zorder=10, 
         label = r'[C II] $\lambda 158 \mu m$')
plt.plot(T,1.0e-27*coolO_10,'-.',color='black',lw=1,zorder=10, 
         label = r'Ly$\alpha$')

# label components

# =============================================================================
# lfs = np.rint(0.8*axisFontSize)
# plt.text(1000.0,1.7e-26,'Total',fontsize=lfs,rotation=10.0,ha='center',va='bottom')
# plt.text(80.0,1.0e-28,r'[O I] $\lambda63\mu m$',fontsize=lfs) #r'$[\textsc{O\,i}]\,\lambda$63$\mu m$'
# plt.text(3000.0,3.5e-27,r'[C II] $\lambda 158 \mu m$',
#          fontsize=lfs,rotation=3.0,ha='center') #r'$[\textsc{C\,ii}]\,\lambda$158$\mu m$'
# plt.text(5400.0,1.0e-28,r'Ly$\alpha$',fontsize=lfs,ha='center')
# =============================================================================
plt.title("Ten Percent O and C abundances")
plt.legend(loc="upper left")

# make the figure

plt.plot()
plt.savefig(plotFile,bbox_inches='tight',facecolor='white')

#%% 1.11 FGH book plot
plotFile = f'Fig1_11.{figFmt}'

fig,ax = plt.subplots()
fig.set_dpi(dpi)
fig.set_size_inches(wInches,hInches,forward=True)

ax.tick_params('both',length=6,width=lwidth,which='major',direction='in',top='on',right='on')
ax.tick_params('both',length=3,width=lwidth,which='minor',direction='in',top='on',right='on')

# Limits

minNe = 0.01     # cm^{-3}
maxNe = 20000.0 

# Labels

xLabel = r'Temperature [K]'
yLabel = r'$n$ [cm$^{-3}$]'

plt.xlim(minT,maxT)
ax.set_xscale('log')
ax.set_xticks([10,100,1000,1.0e4])
ax.set_xticklabels(['10','100','1000','10$^{4}$'])
plt.xlabel(xLabel)

plt.ylim(minNe,maxNe)
ax.set_yscale('log')
ax.set_yticks([0.01,0.1,1.0,10.,100.,1e3,1e4])
ax.set_yticklabels(['0.01','0.1','1','10','100','1000','10$^{4}$'])
plt.ylabel(yLabel)

# Plot neq vs T

plt.plot(T,neq,'-',color='black',lw=2,zorder=10)
plt.fill_between(T,neq,maxNe,facecolor="#eaeaea")

# label regions above and below

lfs = np.rint(1.2*axisFontSize)
plt.text(200.0,0.1,'Net heating',fontsize=lfs,ha='center',zorder=10)
plt.text(1000.0,20.0,'Net cooling',fontsize=lfs,ha='center',zorder=10)
plt.title("Book Plot 1.11")
# make the figure

plt.plot()
plt.savefig(plotFile,bbox_inches='tight',facecolor='white')

#%% 1.11 10%

plotFile = f'Fig1_11_ten_percent.{figFmt}'

fig,ax = plt.subplots()
fig.set_dpi(dpi)
fig.set_size_inches(wInches,hInches,forward=True)

ax.tick_params('both',length=6,width=lwidth,which='major',direction='in',top='on',right='on')
ax.tick_params('both',length=3,width=lwidth,which='minor',direction='in',top='on',right='on')

# Limits

minNe = 0.01     # cm^{-3}
maxNe = 20000.0 

# Labels

xLabel = r'Temperature [K]'
yLabel = r'$n$ [cm$^{-3}$]'

plt.xlim(minT,maxT)
ax.set_xscale('log')
ax.set_xticks([10,100,1000,1.0e4])
ax.set_xticklabels(['10','100','1000','10$^{4}$'])
plt.xlabel(xLabel)

plt.ylim(minNe,maxNe)
ax.set_yscale('log')
ax.set_yticks([0.01,0.1,1.0,10.,100.,1e3,1e4])
ax.set_yticklabels(['0.01','0.1','1','10','100','1000','10$^{4}$'])
plt.ylabel(yLabel)

# Plot neq vs T
neq_10 = gain/coolTot_10 #both gain and coolTot are missing factors of 10^-27 so units are okay
plt.plot(T,neq_10,'-',color='black',lw=2,zorder=10)
plt.fill_between(T,neq_10,maxNe,facecolor="#eaeaea", label = "Net cooling")
plt.fill_between(T, neq_10, minNe, facecolor = "#add8e6", label = "Net heating")
plt.legend(loc="upper right")
plt.title("1.11 w/ ten percent abundances")

# label regions above and below

# =============================================================================
# lfs = np.rint(1.2*axisFontSize)
# plt.text(200.0,0.1,'Net heating',fontsize=lfs,ha='center',zorder=10)
# plt.text(1000.0,20.0,'Net cooling',fontsize=lfs,ha='center',zorder=10)
# =============================================================================

# make the figure

plt.plot()
plt.savefig(plotFile,bbox_inches='tight',facecolor='white')

#%% 1.12 book
plotFile = f'Fig1_12.{figFmt}'

fig,ax = plt.subplots()
fig.set_dpi(dpi)
fig.set_size_inches(wInches,hInches,forward=True)

plt.tick_params('both',length=6,width=lwidth,which='major',direction='in',top='on',right='on')
plt.tick_params('both',length=3,width=lwidth,which='minor',direction='in',top='on',right='on')

# Limits

minNe = 0.02     # cm^{-3}
maxNe = 10000.0 

minP = 4.0e-14   # dyne cm^-2
maxP = 1.0e-11

# Labels

xLabel = r'$n$ [cm$^{-3}$]'
yLabel = r'$P$ [dyne cm$^{-2}$]'

plt.xlim(minNe,maxNe)
plt.xscale('log')
ax.set_xticks([0.1,1.0,10.,1.0e2,1.0e3,1.0e4])
ax.set_xticklabels(['0.1','1.0','10','100','1000','10$^4$'])
plt.xlabel(xLabel)

plt.ylim(minP,maxP)
ax.set_yscale('log')
ax.set_yticks([1.0e-13,1.0e-12,1.0e-11])
ax.set_yticklabels(['10$^{-13}$','10$^{-12}$','10$^{-11}$'])
plt.ylabel(yLabel)

# plot the n-P curve

plt.plot(neq,P,'-',color='black',lw=2,zorder=10)
plt.fill_between(neq,P,maxP,facecolor="#eaeaea")

# FGH stability region - estimate from array using scipy.signal argrelmin() and argrelmax()
# peak-finding functions

iMin = argrelmin(P)[0]
iMax = argrelmax(P)[0]

plt.hlines(P[iMin],minNe,maxNe,color='black',ls='--',lw=0.5)
plt.hlines(P[iMax],minNe,maxNe,color='black',ls='--',lw=0.5)

# Reference pressure, 2e-13 dyne/cm^2

pFGH = 2.0e-13

# The FGH points are at zero crossings of P(n)-fghP.  Find the nearest zero-crossing, then
# fit a line to +/-3 points around it and find the crossing point.  This is dodgy generally
# but we get away with it because the P-n curve is well-behaved.

iFGH = np.where(np.diff(np.sign(P-pFGH)))[0]

nFGH = []
for i in iFGH:
    slope, inter, rVal, pVal, stdErr = stats.linregress(neq[i-3:i+3],P[i-3:i+3]-pFGH)
    xZero = -inter/slope
    nFGH.append(xZero)
    # print(f'n_eq = {xZero:.5e} cm^-3')

lfs = np.rint(1.2*axisFontSize)

plt.plot(nFGH[0],pFGH,color='black',marker='o',ms=8,mfc='black')
plt.text(1.4*nFGH[0],pFGH,'F',fontsize=lfs,va='center',zorder=10)
plt.plot(nFGH[1],pFGH,color='black',marker='o',ms=8,mfc='black')
plt.text(1.4*nFGH[1],pFGH,'G',fontsize=lfs,va='center',zorder=10)
plt.plot(nFGH[2],pFGH,color='black',marker='o',ms=8,mfc='black')
plt.text(1.4*nFGH[2],pFGH,'H',fontsize=lfs,va='center',zorder=10)

plt.text(10.0,1.1*P[iMax],'Net cooling',fontsize=lfs,ha='center',va='bottom',zorder=10)
plt.text(1300.0,pFGH,'Net heating',fontsize=lfs,ha='center',va='center',zorder=10)
plt.title("1.12 Book Plot")
# make the figure
plt.axhline(pFGH, label="reference pressure")

plt.plot()
plt.savefig(plotFile,bbox_inches='tight',facecolor='white')

#%% 1.12 10%
plotFile = f'Fig1_12_ten_percent.{figFmt}'

fig,ax = plt.subplots()
fig.set_dpi(dpi)
fig.set_size_inches(wInches,hInches,forward=True)

plt.tick_params('both',length=6,width=lwidth,which='major',direction='in',
                top='on',right='on')
plt.tick_params('both',length=3,width=lwidth,which='minor',direction='in',
                top='on',right='on')

# Limits

minNe = 0.02     # cm^{-3}
maxNe = 10000.0 

minP = 4.0e-14   # dyne cm^-2
maxP = 1.0e-10

# Labels

xLabel = r'$n$ [cm$^{-3}$]'
yLabel = r'$P$ [dyne cm$^{-2}$]'

plt.xlim(minNe,maxNe)
plt.xscale('log')
ax.set_xticks([0.1,1.0,10.,1.0e2,1.0e3,1.0e4])
ax.set_xticklabels(['0.1','1.0','10','100','1000','10$^4$'])
plt.xlabel(xLabel)

plt.ylim(minP,maxP)
ax.set_yscale('log')
ax.set_yticks([1.0e-13,1.0e-12,1.0e-11])
ax.set_yticklabels(['10$^{-13}$','10$^{-12}$','10$^{-11}$'])
plt.ylabel(yLabel)

# plot the n-P curve
P_10 = neq_10*k*T

plt.plot(neq_10,P_10,'-',color='black',lw=2,zorder=10)
plt.fill_between(neq_10,P_10,maxP,facecolor="#eaeaea", label = "net cool")
plt.fill_between(neq_10,P_10,minP,facecolor="#add8e6", label = 'net heat')


# FGH stability region - estimate from array using scipy.signal argrelmin() and argrelmax()
# peak-finding functions

iMin = argrelmin(P_10)[0]
iMax = argrelmax(P_10)[0]

plt.hlines(P_10[iMin],minNe,maxNe,color='black',ls='--',lw=0.5)
plt.hlines(P_10[iMax],minNe,maxNe,color='black',ls='--',lw=0.5)

# Reference pressure, 2e-13 dyne/cm^2

pFGH = 2.0e-13

# The FGH points are at zero crossings of P(n)-fghP.  Find the nearest zero-crossing, then
# fit a line to +/-3 points around it and find the crossing point.  This is dodgy generally
# but we get away with it because the P-n curve is well-behaved.

iFGH = np.where(np.diff(np.sign(P_10-pFGH)))[0]

nFGH = []
for i in iFGH:
    slope, inter, rVal, pVal, stdErr = stats.linregress(neq_10[i-3:i+3],P_10[i-3:i+3]-pFGH)
    xZero = -inter/slope
    nFGH.append(xZero)
    # print(f'n_eq = {xZero:.5e} cm^-3')

plt.title("1.12 ten percent")
# =============================================================================
# lfs = np.rint(1.2*axisFontSize)
# 
plt.plot(nFGH[0],pFGH,color='black',marker='o',ms=8,mfc='black')
plt.text(1.4*nFGH[0],pFGH,'F',fontsize=lfs,va='center',zorder=10)
plt.axhline(pFGH, label="reference pressure")
# plt.plot(nFGH[1],pFGH,color='black',marker='o',ms=8,mfc='black')
# plt.text(1.4*nFGH[1],pFGH,'G',fontsize=lfs,va='center',zorder=10)
# plt.plot(nFGH[2],pFGH,color='black',marker='o',ms=8,mfc='black')
# plt.text(1.4*nFGH[2],pFGH,'H',fontsize=lfs,va='center',zorder=10)
# 
# plt.text(10.0,1.1*P[iMax],'Net cooling',fontsize=lfs,ha='center',va='bottom',zorder=10)
# plt.text(1300.0,pFGH,'Net heating',fontsize=lfs,ha='center',va='center',zorder=10)
# 
# # make the figure
# =============================================================================
plt.legend()
plt.plot()
plt.savefig(plotFile,bbox_inches='tight',facecolor='white')