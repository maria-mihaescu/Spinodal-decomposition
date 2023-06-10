# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:03:42 2023

@author: Maria Mihaescu
"""
import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt


#Multicomponent homogeneous systems

#ranges for composition, temperature and order parameter
X_B=np.arange(0,1,0.01)      # Composition (chemical order parameter)
T=np.arange(50,1000,50)      # Temperature space
eta=np.arange(-0.5,0.5,0.01) # Order parameter (structural)

#Set the physical constants
k_B=1.38e-23  #Boltzmann's constant
N_A=6.02e+23  #Avogadro's number
R_gas=k_B*N_A #Gas constant



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
    try:
       s = np.where((x > 0) & (~np.isnan(x)), x * np.log(x), np.nan)
    
    except RuntimeWarning:
        # Handle the warning, e.g., assign NaN values
        s = np.nan * np.ones_like(x)
    return s


def test_xlog():
    # Test case 1
    X1 = np.array([0.2, 0.5, 0.8])
    expected_result1 = np.array([-0.32188758, -0.34657359, -0.17851484])
    result1 = xlog(X1)
    assert np.allclose(result1, expected_result1), "Test case 1 failed"

    # Test case 2
    X2 = np.array([-1.2, 0, 1.5, 2.3])
    expected_result2 = np.array([np.nan, np.nan, 0.60819766, 1.91569098])
    result2 = xlog(X2)
    assert np.allclose(result2, expected_result2, equal_nan=True), "Test case 2 failed"

    # Test case 3
    X3 = np.array([])
    expected_result3 = np.array([])
    result3 = xlog(X3)
    assert np.array_equal(result3, expected_result3), "Test case 3 failed"

    print("All tests passed successfully!")
    
def xlog_scal(x):

    """
    function for the fomula x*log(x) if x is a float

    Parameters
    ----------
    x : TYPE : float
        DESCRIPTION.parameter to be entered in the calculation function

    Returns
    -------
    s : TYPE : float
        DESCRIPTION. calculated parameter

    """
    if x<0:
        s=np.nan
    else:
        s= x * np.log(x)
    return s
    
def test_xlog_scal():
    x = 0.5
    expected_result = -0.3466

    result = xlog_scal(x)

    assert np.isclose(round(result, 4), expected_result), "xlog_scal test failed"
 
    
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
    
def test_plot_anim_3d():
    X = np.array([[0, 1], [0, 1]])
    Y = np.array([[0, 0], [1, 1]])
    Z = np.array([[0, 1], [1, 0]])
    xlabel = "X"
    ylabel = "Y"
    zlabel = "Z"
    title = "3D Plot"

    fig = plot_anim_3d(X, Y, Z, xlabel, ylabel, zlabel, title)

    assert isinstance(fig, plt.Figure), "plot_anim_3d test failed"


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
    
def test_plot_2d():
    X = np.array([0, 1, 2, 3, 4])
    G = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    xlabel = "X"
    title = "2D Plot"

    fig = plot_2d(X, G, xlabel, title)

    assert isinstance(fig, plt.Figure), "plot_2d test failed"



def interaction_parameter(Z,
                          diff_eV):

    """
    Function calculating the interaction parameter

    Parameters
    ----------
    Z : TYPE int
        DESCRIPTION. atomic number 
    diff_eV : TYPE float
        DESCRIPTION. fraction of eV difference 

    Returns
    -------
    omega : TYPE float
        DESCRIPTION.In teraction parameter in J 

    """
    omega= -Z*diff_eV*1.6e-19

    return omega

def test_interaction_parameter():
    Z = 10
    diff_eV = 0.5
    expected_result = -8e-18

    result = interaction_parameter(Z, diff_eV)

    assert np.isclose(result, expected_result), "interaction_parameter test failed"


def free_energy_XB_eta(T0,omega):
    
    """
    This function calculates the free energy of a binary alloy in the "quasi-chemical" atomistic model.
    The assumptions taken in the model are :
        - Only nearest-neighbours interactions are taken into account
        - Only the configurational Bragg-Williams-Gorsky entropy is considered
        - Strain energy due to atomic size mismatch and vibrational entropy are neglected
    In the (X_B,eta) space.

    Parameters
    ----------
    T0 : TYPE float 
        DESCRIPTION. Temperature for calculation of free energy surface in (X_B,eta) space
    omega : TYPE float
        DESCRIPTION. Interaction parameter in J 

    Returns
    -------
    X_XB_eta : TYPE numpy.ndarray
        DESCRIPTION. Composition in the (X_B,eta) space 
    eta_XB_eta : TYPE numpy.ndarray
        DESCRIPTION. Structural order parameter eta in the (X_B,eta) space 
    G_XB_eta : TYPE numpy.ndarray
        DESCRIPTION. Gibbs free energy G in the (X_B,eta) space 

    """
    # Functions in (X_B,eta) space for T=T0
    x_BG,etaG = np.meshgrid(X_B,eta)  # defines the (X_B, eta) grid
     
    H0=[]
    S0=[]
    G0=[]
     
    for xbg,etag in zip(x_BG,etaG):
        h0=[]
        s0=[]
        g0=[]
         
        #  Finds the unphysical region in the (X_B, eta) mesh. In fact, abs(eta)
        #  must be lower than X_B and 1-X_B (as stated in the model)
        # Sets enthalpy and entropy to "Not a Number" in the unphysical region, to
        # avoid plotting of non physical values
      
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
             
        H0.append(h0) #Enthalpy
        S0.append(s0) #Entropy
        G0.append(g0) #Gibbs free energy
     
    #rename variable in function of the space 
    X_XB_eta= x_BG
    eta_XB_eta= etaG
    G_XB_eta= np.array(G0)
    return X_XB_eta, eta_XB_eta, G_XB_eta
    
def free_energy_XB_T(T0,omega):
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
    
    xG, yG = np.meshgrid(X_B,T)
    h1=N_A*omega*(np.multiply(xG,(1-xG)))
    ts1=-R_gas*(np.multiply(yG,(xlog(xG)+xlog(1-xG))))
    g1=h1-ts1
    
    #rename variable in function of the space 
    X_XB_T= xG
    T_XB_T= yG
    G_XB_T= g1
    return X_XB_T,T_XB_T,G_XB_T

def free_energy_eta_T(T0,omega):
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
        
    Returns
    -------
    eta_eta_T : TYPE numpy.ndarray
        DESCRIPTION. Structural order parameter eta in the (eta,T) space 
    T_eta_T : TYPE numpy.ndarray
        DESCRIPTION. Temperature T in the (eta,T) space 
    G_eta_T : TYPE numpy.ndarray
        DESCRIPTION. Gibbs free energy G in the (eta,T) space 

    """
    
    X_XB_T,T_XB_T,G_XB_T = free_energy_XB_T(T0,omega)
    #Functions in (eta,T) space for X_B=0.5
    
    eG, TG = np.meshgrid(eta,T)
    
    h2=N_A*omega*(0.25+np.multiply(eG,eG))
    ts2=-R_gas*(np.multiply(T_XB_T,(xlog(0.5+eG)+xlog(0.5-eG))))
    g2=h2-ts2

    #rename variable in function of the space 
    eta_eta_T= eG
    T_eta_T= TG
    G_eta_T= g2
    
    return eta_eta_T, T_eta_T, G_eta_T
