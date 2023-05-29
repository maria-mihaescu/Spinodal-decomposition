# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:03:42 2023

@author: maria
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.pyplot import gca
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.ticker import LinearLocator


import time
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

from matplotlib import cm


def xlog(x):
    if x.all()>0:
        return x*np.log(x)
    else:
        return 0
    
#make the plots in order to do the animation rotating in 3d
def plot_anim_3d(X,Y,Z,xlabel,ylabel,zlabel,title):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    wframe = None
    tstart = time.time()
    for phi in np.linspace(0, 180. / np.pi, 100):
        # If a line collection is already remove it before drawing.
        if wframe:
            wframe.remove()
        # Generate data.
        Z = np.cos(2 * np.pi * X + phi) * (1 - np.hypot(X, Y))
        # Plot the new wireframe and pause briefly before continuing.
        wframe = ax.plot_surface(X, Y, Z,cmap=cm.jet, rstride=2, cstride=2)
        #plt.pause(.001)
    
    print('Average FPS: %f' % (100 / (time.time() - tstart)))
    
#This program calculates the free energy of a binary alloy in the
# so-called "quasi-chemical" atomistic model. Only nearest-neighbours
# interactions are taken into account and only the configurational 
# Bragg-Williams-Gorsky entropy is considered, neglecting e.g. strain energy 
# due to atomic size mismatch and vibrational entropy. 

k_B=1.38e-23  #Boltzmann's constant
N_A=6.02e+23  #Avogadro's number
R_gas=k_B*N_A #Gas constant
omega=-8*0.02*1.6e-19 # interaction parameter: Z=8; fraction of eV difference;
T0=550              # Temperature for calculation of free energy surface in (X_B,eta) space
X_B=np.arange(0,1,0.01)      # Composition (chemical order parameter)
T=np.arange(50,1000,50)      # Temperature space
eta=np.arange(-0.5,0.5,0.01) # Order parameter (structural)


# Functions in (X_B,eta) space for T=T0;

x_BG,etaG=np.meshgrid(X_B,eta)  # defines the (X_B, eta) grid
H0=[]
S0=[]
G0=[]

# Uses the xlog function, that handles the cases where the argument of the
# logarithm would be negative, to calculate entropy
#  Finds the unphysical region in the (X_B, eta) mesh. In fact, abs(eta)
#  must be lower than X_B and 1-X_B (see definitions in the model)
# Sets enthalpy and entropy to "Not a Number" in the unphysical region, to
# avoid plotting of non physical values


for xbg,etag in zip(x_BG,etaG):
    h0=[N_A*omega*(x*(1-x)+e**2) for x,e in zip(xbg,etag) if (abs(e)< x) or (abs(e)<(1-x))]
    H0.append(h0) #  Enthalpy
    s0=[-(R_gas/2)*(xlog(x+e)+xlog(x-e)+ xlog(1-x+e)+xlog(1-x-e))
        for x,e in zip(xbg,etag) if (abs(e)< x) or (abs(e)<(1-x))]
    S0.append(s0)
    g0=[h-T0*s for h,s in zip(h0,s0)]
    G0.append(g0)
    
print('H0 =',H0)


# Functions in (X_B,T) space for eta=0

H1=[]
TS1=[]
G1=[]
xG, yG = np.meshgrid(X_B,T)

for xg,yg in zip(xG,yG):
    h1=[N_A*omega*(x*(1-x)) for x in xg]
    H1.append(h1) #  Enthalpy
    
    ts1=[-R_gas*(y*(xlog(x)+xlog(1-x))) for x,y in zip(xg,yg)]
    TS1.append(ts1) #entropy
    
    g1=[h-ts for h,ts in zip(h1,ts1)]
    G1.append(g1) #Gibbs freee energy
    
#Functions in (eta,T) space for X_B=0.5

eG, TG = np.meshgrid(eta,T)
H2=[]
TS2=[]
G2=[]

for eg,tg, yg in zip(eG,TG,yG):
    h2=[N_A*omega*(0.25+e**2) for e in eg]
    H2.append(h2) #  Enthalpy
    
    ts2=[-R_gas*(y*(xlog(0.5+e)+xlog(0.5-e))) for e,y in zip(eg,yg)]
    TS2.append(ts2) #entropy
    
    g2=[h-ts for h,ts in zip(h2,ts2)]
    G2.append(g2) #Gibbs freee energy
    
# GRAPHICS ===============================================================

# Free energy vs composition at different T for order parameter eta=0

plt.figure()
for g1 in G1:
    plt.plot(X_B,g1)
plt.xlabel('X_B')
plt.ylabel('G [J/mole]')
plt.title('G vs X_B for different T, eta=0 , N_A*Omega={:.2f} J'.format(N_A*omega))
plt.show()


# Free energy vs order paramer eta for different T at X_B=0.5

plt.figure()
for g2 in G2:
    plt.plot(eta,g2)
    
plt.xlabel('eta')
plt.ylabel('G [J/mole]')
plt.title('G vs eta for different T, X_B=0.5 , N_A*Omega={:.2f} J'.format(N_A*omega))
plt.show()


# Free energy surface in (X_B,eta) space for T=T0 
plot_anim_3d(x_BG,etaG,G0,
             'X_B','eta','G [ J/mole ]'
             ,'G vs X_B and eta, N_A\Omega={:.2f} J , T={:.2f}'.format(N_A*omega,T0))

# Free energy surface in (X_B,T) space for order parameter eta=0 
plot_anim_3d(xG,yG,G1,
             'X_B','T [K]','G [ J/mole ]'
             ,'G vs X_B and T, eta=0, N_A\Omega={:.2f} J'.format(N_A*omega))

#Free energy surface in (eta,T) space for equimolar composition (X_B=0.5)
plot_anim_3d(eG,TG,G2,
             'eta','T [K]','G [ J/mole ]'
             ,'G vs eta and T, X_B=0.5, N_A\Omega={:.2f} J'.format(N_A*omega))

