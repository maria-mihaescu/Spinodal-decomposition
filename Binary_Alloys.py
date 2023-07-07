# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:03:42 2023

@author: Maria Mihaescu
"""
import numpy as np
import matplotlib
from matplotlib import cm

#Set the physical constants

k_B=1.38e-23  #Boltzmann's constant
N_A=6.02e+23  #Avogadro's number
R_gas=k_B*N_A #Gas constant



def ranges(X_B_min,
           X_B_max,
           X_B_step,
           T_min,
           T_max,
           T_step,
           eta_min,
           eta_max,
           eta_step):
    """
    

    Parameters
    ----------
    X_B_min : TYPE float
        DESCRIPTION. minimum of the composition range
    X_B_max : TYPE float
        DESCRIPTION. maximum of the composition range
    X_B_step : TYPE float
        DESCRIPTION. step of the composiion range
    T_min : TYPE float 
        DESCRIPTION. minimum of the temperature range
    T_max : TYPE float 
        DESCRIPTION.maximum of the temperature range
    T_step : TYPE float 
        DESCRIPTION.step of the temperature range
    eta_min : TYPE float
        DESCRIPTION. minimum of the structural order parameter
    eta_max : TYPE float
        DESCRIPTION. maximum of the structural order parameter
    eta_step : TYPE float
        DESCRIPTION.step of the structural order parameter

    Returns
    -------
    X_B : TYPE array
        DESCRIPTION. composition range
    T : TYPE array
        DESCRIPTION. temperature range
    eta : TYPE array 
        DESCRIPTION. structural order parameter range 

    """
    #ranges for composition, temperature and order parameter
    X_B=np.arange(X_B_min,X_B_max,X_B_step)      # Composition (chemical order parameter)
    T=np.arange(T_min,T_max,T_step)      # Temperature space
    eta=np.arange(eta_min,eta_max,eta_step) # Order parameter (structural)
    
    return X_B,T,eta


def xlog(x):
    
    """
    function for the fomula x*log(x) if x is an array

    Parameters
    ----------
    x : TYPE nd.ndarray
        DESCRIPTION. parameter to be entered in the calculation function

    Returns
    -------
    s : TYPE nd.ndarray
        DESCRIPTION. calculated array  

    """
    #condition to suppress the runtime warning. 
    #This ensures that the calculation proceeds without raising an exception and replaces the undefined logarithmic values with NaN
    
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where((x > 0) & (~np.isnan(x)), x * np.log(x), np.nan)   
    return s


    
    
def plot_anim_3d(X,
                 Y,
                 Z,
                 xlabel,
                 ylabel,
                 zlabel,
                 title):

    """function to do plots that rotate in 3D

    Parameters
    ----------
    X : TYPE : numpy.ndarray
        DESCRIPTION. X data in the space (X,Y)
    Y : TYPE : numpy.ndarray
        DESCRIPTION. Y data in the space (X,Y)
    Z : TYPE : numpy.ndarray
        DESCRIPTION. Z data in function of the X,Y data 
    xlabel : TYPE : string
        DESCRIPTION: Label of the x axis 
    ylabel : TYPE : string 
        DESCRIPTION. Label of the y axis
    zlabel : TYPE string
        DESCRIPTION. Label of the z axis
    title : TYPE string
        DESCRIPTION. title of the graph 

    Returns
    -------
    fig : TYPE matplotlib.figure.Figure()
        DESCRIPTION. 3D figure 

    """
    fig=matplotlib.figure.Figure()
    ax = fig.add_subplot(projection='3d')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    wframe = None
    for phi in np.linspace(0, 180. / np.pi, 100):
        # If a line collection is already remove it before drawing.
        if wframe:
            wframe.remove()
        
        # Plot the new surface plot
        wframe = ax.plot_surface(X, Y, Z,cmap=cm.nipy_spectral_r, rstride=2, cstride=2)
    
    #print a line to show that the 3D plot is done
    print("3D Plot done !")
    return fig
    


def plot_2d(X,
            G,
            xlabel,
            title) :

    """
    
    Function that returns a 2D plot of the free energy in function of an order parameter 
    
    Parameters
    ----------
    X : TYPE numpy.ndarray
        DESCRIPTION. Range for the order parameter 
    G : TYPE numpy.ndarray
        DESCRIPTION. Calculated free energy for this order parameter 
    xlabel : TYPE string
        DESCRIPTION. Label of the xaxis
    title : TYPE string
        DESCRIPTION. title of the plot

    Returns
    -------
    fig : TYPE matplotlib.figure.Figure()
        DESCRIPTION. Figure of the free energy in function of an order parameter

    """
    fig=matplotlib.figure.Figure()
    ax = fig.add_subplot()
    for g in G:
        ax.plot(X,g)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('G [J/mole]')
        ax.set_title(title)
    return fig
    


def interaction_parameter(Z,
                          diff_eV):

    """
    Function calculating the interaction parameter

    Parameters
    ----------
    Z : TYPE int
        DESCRIPTION. number of nearest neighbours
    diff_eV : TYPE float
        DESCRIPTION. fraction of eV difference 

    Returns
    -------
    omega : TYPE float
        DESCRIPTION.In teraction parameter in J 

    """
    conversion_factor= 1.6e-19 #conversion factor from eV to J
    omega= -Z*diff_eV*conversion_factor

    return omega


def enthalpy_scal(eta,
                  composition,
                  omega,
                  T0):
    """
    function that calculates the enthalpy and finds the unphysical region in the 
    (X_B, eta) mesh. As abs(eta) must be lower than X_B and 1-X_B (as stated in the model).
    The enthalpy is set to "Not a Number" in the unphysical region, to avoid plotting of non physical values

    Parameters
    ----------
    eta : TYPE float
          DESCRIPTION. Order parameter (structural)
    composition : TYPE float
                  DESCRIPTION. composition [atomic fraction]
    omega : TYPE float
            DESCRIPTION.In teraction parameter in J 
    T0 : TYPE float
        DESCRIPTION. temperature in K

    Returns
    -------
    h : TYPE float
        DESCRIPTION. enthalpy

    """
    x=composition
    e=eta
    
    if abs(e)>=x or abs(e)>=(1-x):
        h=np.nan
        
    else:
        
        h=N_A*omega*(x*(1-x)+e**2)
        
    return h




def entropie_scal(eta,
                  composition,
                  omega,
                  T0):
    
    """
    function that calculates the entropie and finds the unphysical region in the 
    (X_B, eta) mesh. As abs(eta) must be lower than X_B and 1-X_B (as stated in the model).
    The entropy is set to "Not a Number" in the unphysical region, to avoid plotting of non physical values

    Parameters
    ----------
    eta : TYPE float
          DESCRIPTION. Order parameter (structural)
    composition : TYPE float
                  DESCRIPTION. composition [atomic fraction]
    omega : TYPE float
            DESCRIPTION.In teraction parameter in J 
    T0 : TYPE float
        DESCRIPTION. temperature in K

    Returns
    -------
    s : TYPE float
        DESCRIPTION. entropie

    """
    x=composition
    e=eta
    
    if abs(e)>=x or abs(e)>=(1-x):
        s=np.nan
    else:
        s=-(R_gas/2)*(xlog(x+e)+xlog(x-e)+ xlog(1-x+e)+xlog(1-x-e))
    return s



def free_energy_XB_eta(T0, omega, X_B, eta):
    
    """
    This function calculates the free energy of a binary alloy in the "quasi-chemical" atomistic model.
    The assumptions taken in the model are:
        - Only nearest-neighbours interactions are taken into account
        - Only the configurational Bragg-Williams-Gorsky entropy is considered
        - Strain energy due to atomic size mismatch and vibrational entropy are neglected
    In the (X_B,eta) space.

    Parameters
    ----------
    T0 : float
        Temperature for calculation of free energy surface in (X_B,eta) space
    omega : float
        Interaction parameter in J
    X_B : numpy.ndarray
        Composition range
    eta : numpy.ndarray
        Structural order parameter range

    Returns
    -------
    X_XB_eta : numpy.ndarray
        Composition in the (X_B,eta) space
    eta_XB_eta : numpy.ndarray
        Structural order parameter eta in the (X_B,eta) space
    G_XB_eta : numpy.ndarray
        Gibbs free energy G in the (X_B,eta) space
    """

    # Create meshgrid in (X_B,eta) space
    X_XB_eta, eta_XB_eta = np.meshgrid(X_B, eta)

    # Calculate enthalpy, entropy, and free energy for each (X_B,eta) pair
    H_XB_eta = np.zeros_like(X_XB_eta)
    S_XB_eta = np.zeros_like(X_XB_eta)
    G_XB_eta = np.zeros_like(X_XB_eta)

    for i in range(X_XB_eta.shape[0]):
        for j in range(X_XB_eta.shape[1]):
            
            x = X_XB_eta[i, j]
            e = eta_XB_eta[i, j]

            h = enthalpy_scal(e, x, omega, T0)
            s = entropie_scal(e, x, omega, T0)

            H_XB_eta[i, j] = h
            S_XB_eta[i, j] = s
            G_XB_eta[i, j] = h - T0 * s

    return X_XB_eta, eta_XB_eta, G_XB_eta


    
def enthalpie_XB_T(X_XB_T,
                   T0,
                   omega):
    
    """
    function to caculate the enthalpie in the (XB,T) space 

    Parameters
    ----------
    X_XB_T : TYPE array
        DESCRIPTION. Composition in the (XB,T) space
    T0 : TYPE float 
        DESCRIPTION. Temperature for calculation of free energy surface in (X_B,eta) space
    omega : TYPE float
        DESCRIPTION. Interaction parameter in J 


    Returns
    -------
    h : TYPE array
        DESCRIPTION. enthalpie in the (XB,T) space 

    """
    
    h= N_A*omega*(np.multiply(X_XB_T,(1-X_XB_T)))
    
    return h



def T_entropie_XB_T(X_XB_T, T_XB_T,T0,omega):
    
    """
    function to calculate the temperature times entropie in the (XB,T) space

    Parameters
    ----------
    X_XB_T : TYPE array
        DESCRIPTION. composition in the (X_B,T) space
    T_XB_T : TYPE array
        DESCRIPTION. temperature in [K] in the (X_B,T) space
    T0 : TYPE float 
        DESCRIPTION. Temperature for calculation of free energy surface in (X_B,eta) space
    omega : TYPE float
        DESCRIPTION. Interaction parameter in J 

    Returns
    -------
    ts : TYPE array
        DESCRIPTION. temperature*Entropie

    """
    
    ts=-R_gas*(np.multiply(T_XB_T,(xlog(X_XB_T)+xlog(1-X_XB_T))))
    
    return ts



def free_energy_XB_T(T0,
                     omega,
                     X_B,
                     T):
    
    """
    This function calculates the free energy of a binary alloy in the "quasi-chemical" atomistic model.
    The assumptions taken in the model are :
        - Only nearest-neighbours interactions are taken into account
        - Only the configurational Bragg-Williams-Gorsky entropy is considered
        - Strain energy due to atomic size mismatch and vibrational entropy are neglected
    In the (X_B,T) space.

    Parameters
    ----------
    T0 : TYPE float 
        DESCRIPTION. Temperature for calculation of free energy surface in (X_B,eta) space
    omega : TYPE float
        DESCRIPTION. Interaction parameter in J 
    X_B : TYPE array
        DESCRIPTION. composition range
    T : TYPE array
        DESCRIPTION. temperature range    
    Returns
    -------

    X_XB_T : TYPE numpy.ndarray
        DESCRIPTION. Composition X in the (X_B,T) space 
    T_XB_T : TYPE numpy.ndarray
        DESCRIPTION. Temperature T in the (X_B,T) space 
    G_XB_T : TYPE numpy.ndarray
        DESCRIPTION. Gibbs free energy G in the (X_B,T) space 

    """
    # Functions in (X_B,T) space for eta=0
    X_XB_T, T_XB_T = np.meshgrid(X_B,T)
    
    h1=enthalpie_XB_T(X_XB_T,T0,omega)
    ts1=T_entropie_XB_T(X_XB_T, T_XB_T,T0,omega)
    G_XB_T=h1-ts1
    
    return X_XB_T,T_XB_T,G_XB_T




def enthalpie_eta_T(eta_eta_T,T0,omega):
    
    """
    function to caculate the enthalpie in the (XB,T) space 

    Parameters
    ----------
    eta_eta_T : TYPE array
        DESCRIPTION. structural order parameter in the (eta,T) space
    T0 : TYPE float 
        DESCRIPTION. Temperature for calculation of free energy surface in (X_B,eta) space
    omega : TYPE float
        DESCRIPTION. Interaction parameter in J 


    Returns
    -------
    h : TYPE array
        DESCRIPTION. enthalpie in the (XB,T) space 

    """
    
    h= N_A*omega*(0.25+np.multiply(eta_eta_T,eta_eta_T))
    
    return h



def T_entropie_eta_T(eta_eta_T,T_XB_T,T0,omega):
    
    """
    function to calculate the temperature times entropie in the (eta,T) space

    Parameters
    ----------
    eta_eta_T : TYPE array
        DESCRIPTION. structural order parameter in the (eta,T) space
    T_XB_T : TYPE numpy.ndarray
        DESCRIPTION. Temperature T in the (X_B,T) space 
    T0 : TYPE float 
        DESCRIPTION. Temperature for calculation of free energy surface in (X_B,eta) space
    omega : TYPE float
        DESCRIPTION. Interaction parameter in J 

    Returns
    -------
    ts : TYPE array
        DESCRIPTION. temperature*Entropie

    """
    
    ts=-R_gas*(np.multiply(T_XB_T,(xlog(0.5+eta_eta_T)+xlog(0.5-eta_eta_T))))
    
    return ts




def free_energy_eta_T(T0,
                      omega,
                      X_B,
                      eta,
                      T):
    
    """
    This function calculates the free energy of a binary alloy in the "quasi-chemical" atomistic model.
    The assumptions taken in the model are :
        - Only nearest-neighbours interactions are taken into account
        - Only the configurational Bragg-Williams-Gorsky entropy is considered
        - Strain energy due to atomic size mismatch and vibrational entropy are neglected
    In the (eta,T) space.

    Parameters
    ----------
    T0 : TYPE float 
        DESCRIPTION. Temperature for calculation of free energy surface in (X_B,eta) space
    omega : TYPE float
        DESCRIPTION. Interaction parameter in J 
    X_B : TYPE array
        DESCRIPTION. composition range
    eta : TYPE array
        DESCRIPTION. structural order parameter range
    T : TYPE array
        DESCRIPTION. temperature range
        
    Returns
    -------
    eta_eta_T : TYPE numpy.ndarray
        DESCRIPTION. Structural order parameter eta in the (eta,T) space 
    T_eta_T : TYPE numpy.ndarray
        DESCRIPTION. Temperature T in the (eta,T) space 
    G_eta_T : TYPE numpy.ndarray
        DESCRIPTION. Gibbs free energy G in the (eta,T) space 

    """
    
    X_XB_T,T_XB_T,G_XB_T = free_energy_XB_T(T0,omega,X_B,T)
    
    #Functions in (eta,T) space for X_B=0.5
    
    eta_eta_T, T_eta_T = np.meshgrid(eta,T)
    
    h2=enthalpie_eta_T(eta_eta_T, T0, omega)
    ts2=T_entropie_eta_T(eta_eta_T, T_XB_T, T0, omega)
    G_eta_T=h2-ts2
    
    return eta_eta_T, T_eta_T, G_eta_T


