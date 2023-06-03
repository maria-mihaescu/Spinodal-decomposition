# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:03:42 2023

@author: Maria Mihaescu
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib


def xlog(x):
    if x.all()<0:
        s=np.nan
    else:
        s= np.multiply(x,np.log(x))
    return s

def xlog_scal(x):
    if x<0:
        s=np.nan
    else:
        s= x * np.log(x)
    return s
    
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
        
        # Plot the new surface plot
        wframe = ax.plot_surface(X, Y, Z,cmap=cm.nipy_spectral_r, rstride=2, cstride=2)
    
    print('Average FPS: %f' % (100 / (time.time() - tstart)))
    
    
def plot_2d(X,G,xlabel,title) :
    fig, ax = plt.subplots()
    for g in G:
        ax.plot(X,g)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('G [J/mole]')
        ax.set_title(title)
        fig.show()
    
#This program calculates the free energy of a binary alloy in the
# so-called "quasi-chemical" atomistic model. Only nearest-neighbours
# interactions are taken into account and only the configurational 
# Bragg-Williams-Gorsky entropy is considered, neglecting e.g. strain energy 
# due to atomic size mismatch and vibrational entropy. 

#Set the constants
k_B=1.38e-23  #Boltzmann's constant
N_A=6.02e+23  #Avogadro's number
R_gas=k_B*N_A #Gas constant


def interaction_parameter(Z,diff_eV):
    # interaction parameter: 
    #Z= atomic number ;
    #diff_eV = fraction of eV difference
    omega= -Z*diff_eV*1.6e-19
    return omega


def set_free_energy(X_B,eta,T,T0,omega):
    
    # Functions in (X_B,eta) space for T=T0;

    # Uses the xlog function, that handles the cases where the argument of the
    # logarithm would be negative, to calculate entropy
    
    #  Finds the unphysical region in the (X_B, eta) mesh. In fact, abs(eta)
    #  must be lower than X_B and 1-X_B (see definitions in the model)
    
    # Sets enthalpy and entropy to "Not a Number" in the unphysical region, to
    # avoid plotting of non physical values
    
    x_BG,etaG = np.meshgrid(X_B,eta)  # defines the (X_B, eta) grid
    
    H0=[]
    S0=[]
    G0=[]
    
    for xbg,etag in zip(x_BG,etaG):
        h0=[]
        s0=[]
        g0=[]
        
        for x,e in zip(xbg,etag):
            if abs(e)>=x or abs(e)>=(1-x):
                h=np.nan
                s=np.nan
                g=np.nan
                
            else:
                h=N_A*omega*(x*(1-x)+e**2)
                s=-(R_gas/2)*(xlog_scal(x+e)+xlog_scal(x-e)+ xlog_scal(1-x+e)+xlog_scal(1-x-e))
                g=h-T0*s
                
            h0.append(h)
            s0.append(s)
            g0.append(g)
            
        H0.append(h0) #  Enthalpy
        S0.append(s0) #Entropy
        G0.append(g0) #Gibbs free energy
    
    #rename variable in function of the space 
    X_XB_eta= x_BG
    eta_XB_eta= etaG
    G_XB_eta= np.array(G0)

    # Functions in (X_B,T) space for eta=0
    
    xG, yG = np.meshgrid(X_B,T)
    h1=N_A*omega*(np.multiply(xG,(1-xG)))
    ts1=-R_gas*(np.multiply(yG,(xlog(xG)+xlog(1-xG))))
    g1=h1-ts1
    
    #rename variable in function of the space 
    X_XB_T= xG
    T_XB_T= yG
    G_XB_T= g1

    
    #Functions in (eta,T) space for X_B=0.5
    
    eG, TG = np.meshgrid(eta,T)
    
    h2=N_A*omega*(0.25+np.multiply(eG,eG))
    ts2=-R_gas*(np.multiply(T_XB_T,(xlog(0.5+eG)+xlog(0.5-eG))))
    g2=h2-ts2

    #rename variable in function of the space 
    eta_eta_T= eG
    T_eta_T= TG
    G_eta_T= g2
    
    return X_XB_eta,eta_XB_eta,G_XB_eta,X_XB_T,T_XB_T,G_XB_T,eta_eta_T,T_eta_T,G_eta_T


#Set parameters of the material:
    
#parameters used in the material
Z=8
diff_eV=0.02
T0=550     # Temperature for calculation of free energy surface in (X_B,eta) space

#ranges for composition, temperature and order parameter
X_B=np.arange(0,1,0.01)      # Composition (chemical order parameter)
T=np.arange(50,1000,50)      # Temperature space
eta=np.arange(-0.5,0.5,0.01) # Order parameter (structural)

omega = interaction_parameter(Z,diff_eV)
#Set parameters for the graphs
X_XB_eta,eta_XB_eta,G_XB_eta,X_XB_T,T_XB_T,G_XB_T,eta_eta_T,T_eta_T,G_eta_T = set_free_energy(X_B,eta,T,T0,omega)


# Free energy vs composition at different T for order parameter eta=0

plot_2d(X_B,G_XB_T,'X_B','G vs X_B for different T, eta=0')

# Free energy vs order paramer eta for different T at X_B=0.5
plot_2d(eta,G_eta_T,'eta','G vs eta for different T, X_B=0.5')


#Free energy surface in the (X_B,eta) space for temperature T=T0
plot_anim_3d(X_XB_eta,eta_XB_eta,G_XB_eta,
             'X_B','eta','G [ J/mole ]'
             ,'G vs X_B and eta, N_A\Omega={:.2f} J , T={:.2f}'.format(N_A*omega,T0))


# Free energy surface in (X_B,T) space for order parameter eta=0 
plot_anim_3d(X_XB_T,T_XB_T,G_XB_T,
             'X_B','T [K]','G [ J/mole ]'
             ,'G vs X_B and T, eta=0, N_A\Omega={:.2f} J'.format(N_A*omega))

#Free energy surface in (eta,T) space for equimolar composition (X_B=0.5)
plot_anim_3d(eta_eta_T,T_eta_T,G_eta_T,
             'eta','T [K]','G [ J/mole ]'
             ,'G vs eta and T, X_B=0.5, N_A\Omega={:.2f} J'.format(N_A*omega))

