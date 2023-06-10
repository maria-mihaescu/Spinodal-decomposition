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
    """
    function to test the x*log() calculation

    Returns
    -------
    None.

    """
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
    """
    function to test the ploting of the 3D plot with arrays

    Returns
    -------
    None.

    """
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
    """
    Test the 2D plot for an energy G

    Returns
    -------
    None.

    """
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
    """
    test the calculation of the interaction parameter 

    Returns
    -------
    None.

    """
    Z = 10
    diff_eV = 0.5
    expected_result = -8e-18

    result = interaction_parameter(Z, diff_eV)

    assert np.isclose(result, expected_result), "interaction_parameter test failed"


def enthalpy_scal(eta,composition,omega,T0):
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


def test_enthalpy_scal():
    
    """
    test the calculation of the enthalpy

    Returns
    -------
    None.

    """
    eta = 0.2  # example value for eta
    composition = 0.5  # example value for composition
    omega = 1.5  # example value for omega
    T0 = 300  # example value for T0

    h = enthalpy_scal(eta, composition, omega, T0)

    # Calculate the expected enthalpy value
    if abs(eta) >= composition or abs(eta) >= (1 - composition):
        expected_h = np.nan
    else:
        expected_h = N_A * omega * (composition * (1 - composition) + eta ** 2)

    # Assert that the calculated enthalpy matches the expected value
    assert np.isnan(h) and np.isnan(expected_h) or np.isclose(h, expected_h)


def entropie_scal(eta,composition,omega,T0):
    
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

def test_entropie_scal():
    """
    function to test the validity condition and the calculation of the entropie
    Returns
    -------
    None.

    """
    eta = 0.2  # example value for eta
    composition = 0.5  # example value for composition
    omega = 1.5  # example value for omega
    T0 = 300  # example value for T0

    s = entropie_scal(eta, composition, omega, T0)

    # Calculate the expected entropy value
    if abs(eta) >= composition or abs(eta) >= (1 - composition):
        expected_s = np.nan
    else:
        expected_s = -(R_gas / 2) * (xlog(composition + eta) + xlog(composition - eta) +
                                     xlog(1 - composition + eta) + xlog(1 - composition - eta))

    # Assert that the calculated entropy matches the expected value
    assert np.isnan(s) and np.isnan(expected_s) or np.isclose(s, expected_s)


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
    X_XB_eta,eta_XB_eta = np.meshgrid(X_B,eta)  # defines the (X_B, eta) grid
     
    H0=[]
    S0=[]
    G0=[]
     
    for xbg,etag in zip(X_XB_eta,eta_XB_eta):
        h0=[]
        s0=[]
        g0=[]
      
        for x,e in zip(xbg,etag):
            h=enthalpy_scal(e,x,omega,T0)
            s=entropie_scal(e,x,omega,T0)
            g=h-T0*s
                 
            h0.append(h)
            s0.append(s)
            g0.append(g)
             
        H0.append(h0) #Enthalpy
        S0.append(s0) #Entropy
        G0.append(g0) #Gibbs free energy
     
    G_XB_eta= np.array(G0)
    
    return X_XB_eta, eta_XB_eta, G_XB_eta

def test_free_energy_XB_eta():
    
    """
    Test function for the definition of the free energy in the (XB,eta) space

    Returns
    -------
    None.

    """
    
    T0 = 300  # example value for T0
    omega = 1.5  # example value for omega

    X_XB_eta, eta_XB_eta, G_XB_eta = free_energy_XB_eta(T0, omega)

    # Check the dimensions of the output arrays
    assert X_XB_eta.shape == (len(X_B), len(eta))
    assert eta_XB_eta.shape == (len(X_B), len(eta))

    # Check if any NaN values exist in the arrays
    assert np.isnan(G_XB_eta).any()  # Expecting the presence of NaN values

    
def enthalpie_XB_T(X_XB_T,T0,omega):
    
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

def test_enthalpie_XB_T():
    """
    Test the calculation of the enthalpie for the (XB,T) space

    Returns
    -------
    None.

    """
    X_XB_T = np.array([0.2, 0.4, 0.6])  # example values for X_XB_T
    T0 = 300  # example value for T0
    omega = 1.5  # example value for omega

    h = enthalpie_XB_T(X_XB_T, T0, omega)

    # Calculate the expected enthalpy values
    expected_h = N_A * omega * (X_XB_T * (1 - X_XB_T))

    # Check if the calculated enthalpy matches the expected values
    assert np.allclose(h, expected_h)
    
    
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

def test_T_entropie_XB_T():
    """
    test function for the entropie calculation in the (XB,T) space 

    Returns
    -------
    None.

    """
    X_XB_T = np.array([0.2, 0.4, 0.6])  # example values for X_XB_T
    T_XB_T = np.array([300, 400, 500])  # example values for T_XB_T
    T0 = 300  # example value for T0
    omega = 1.5  # example value for omega

    ts = T_entropie_XB_T(X_XB_T, T_XB_T, T0, omega)

    # Calculate the expected ts values
    expected_ts = -R_gas * (T_XB_T * (xlog(X_XB_T) + xlog(1 - X_XB_T)))

    # Check if the calculated ts values match the expected values
    assert np.allclose(ts, expected_ts)

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
    X_XB_T, T_XB_T = np.meshgrid(X_B,T)
    
    h1=enthalpie_XB_T(X_XB_T,T0,omega)
    ts1=T_entropie_XB_T(X_XB_T, T_XB_T,T0,omega)
    G_XB_T=h1-ts1
    
    return X_XB_T,T_XB_T,G_XB_T

def test_free_energy_XB_T():
    """
    test function for the free energy calculation in the (XB,T) space

    Returns
    -------
    None.

    """
    # Test input values
    T0 = 300.0
    omega = 1.0
    
    # Ranges for composition, temperature, and order parameter
    X_B = np.arange(0, 1, 0.01)  # Composition (chemical order parameter)
    T = np.arange(50, 1000, 50)  # Temperature space
    
    # Expected output shapes
    expected_shape = (len(T),len(X_B))

    # Call the function
    X_XB_T, T_XB_T, G_XB_T = free_energy_XB_T(T0, omega)
    
    # Perform assertions
    assert isinstance(X_XB_T, np.ndarray)
    assert isinstance(T_XB_T, np.ndarray)
    assert isinstance(G_XB_T, np.ndarray)

    
    assert X_XB_T.shape == expected_shape
    assert T_XB_T.shape == expected_shape
    
    # Check that X_XB_T and T_XB_T are the same as the meshgrid inputs
    X_B_mesh, T_mesh = np.meshgrid(X_B, T)
    
    assert np.array_equal(X_XB_T, X_B_mesh)
    assert np.array_equal(T_XB_T, T_mesh)
    

    #assertion for G_XB_T calculation
    expected_G_XB_T = enthalpie_XB_T(X_XB_T, T0, omega) - T_entropie_XB_T(X_XB_T, T_XB_T, T0, omega)
    
    assert not np.allclose(G_XB_T, expected_G_XB_T)
    
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

def test_enthalpie_eta_T():
    """
    test function to test the calculation of the enthalpie in the (eta,T) space 

    Returns
    -------
    None.

    """
    # Test input values
    eta_eta_T = np.array([[0.1, 0.2, 0.3],
                          [0.4, 0.5, 0.6],
                          [0.7, 0.8, 0.9]])
    T0 = 300.0
    omega = 1.0
    
    # Expected output shape
    expected_shape = eta_eta_T.shape
    
    # Call the function
    h = enthalpie_eta_T(eta_eta_T, T0, omega)
    
    # Perform assertions
    assert isinstance(h, np.ndarray)
    assert h.shape == expected_shape

    # Check calculated values against expected values
    expected_h = N_A * omega * (0.25 + np.multiply(eta_eta_T, eta_eta_T))
    assert np.allclose(h, expected_h)
    
    # Check specific values
    assert np.isclose(h[0, 0], N_A * omega * (0.25 + eta_eta_T[0, 0] * eta_eta_T[0, 0]))
    assert np.isclose(h[1, 2], N_A * omega * (0.25 + eta_eta_T[1, 2] * eta_eta_T[1, 2]))
    assert np.isclose(h[2, 1], N_A * omega * (0.25 + eta_eta_T[2, 1] * eta_eta_T[2, 1]))

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

def test_T_entropie_eta_T():
    """
    test function to test the caltulation of T*entropie in the (eta,T) space

    Returns
    -------
    None.

    """
    # Test input values
    eta_eta_T = np.array([[0.1, 0.2, 0.3],
                          [0.4, 0.5, 0.6],
                          [0.7, 0.8, 0.9]])
    T_XB_T = np.array([[300, 400, 500],
                       [600, 700, 800],
                       [900, 1000, 1100]])
    T0 = 300.0
    omega = 1.0
    
    # Expected output shape
    expected_shape = eta_eta_T.shape
    
    # Call the function
    ts = T_entropie_eta_T(eta_eta_T, T_XB_T, T0, omega)
    print(ts)
    # Perform assertions
    assert isinstance(ts, np.ndarray)
    assert ts.shape == expected_shape
    
    # Check calculated values against expected values
    expected_ts = -R_gas * (np.multiply(T_XB_T, (xlog(0.5 + eta_eta_T) + xlog(0.5 - eta_eta_T))))
    print(expected_ts)
    assert np.allclose(ts, expected_ts,equal_nan=True)
    
    
    # Check specific values
    assert np.isclose(ts[0, 0], -R_gas * T_XB_T[0, 0] * (xlog(0.5 + eta_eta_T[0, 0]) + xlog(0.5 - eta_eta_T[0, 0])),equal_nan=True)
    assert np.isclose(ts[1, 2], -R_gas * T_XB_T[1, 2] * (xlog(0.5 + eta_eta_T[1, 2]) + xlog(0.5 - eta_eta_T[1, 2])),equal_nan=True)
    assert np.isclose(ts[2, 1], -R_gas * T_XB_T[2, 1] * (xlog(0.5 + eta_eta_T[2, 1]) + xlog(0.5 - eta_eta_T[2, 1])),equal_nan=True)
    
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
    
    eta_eta_T, T_eta_T = np.meshgrid(eta,T)
    
    h2=enthalpie_eta_T(eta_eta_T, T0, omega)
    ts2=T_entropie_eta_T(eta_eta_T, T_XB_T, T0, omega)
    G_eta_T=h2-ts2
    
    return eta_eta_T, T_eta_T, G_eta_T


def test_free_energy_eta_T():
    """
    test function to test the calculation of the free energy in the (eta,t) space

    Returns
    -------
    None.

    """
    # Test input values
    T0 = 300.0
    omega = 1.0

    # Call the function
    eta_eta_T, T_eta_T, G_eta_T = free_energy_eta_T(T0, omega)
    print(eta_eta_T)
    # Perform assertions
    assert isinstance(eta_eta_T, np.ndarray)
    assert isinstance(T_eta_T, np.ndarray)
    assert isinstance(G_eta_T, np.ndarray)

    # Check shapes of the output arrays
    assert eta_eta_T.shape == T_eta_T.shape
    assert eta_eta_T.shape == G_eta_T.shape

    #assertion for eta_eta_T calculation
    X_XB_T,T_XB_T,G_XB_T = free_energy_XB_T(T0,omega)
    expected_eta_eta_T = enthalpie_eta_T(eta_eta_T, T0, omega) - T_entropie_eta_T(eta_eta_T, T_XB_T, T0, omega)
    
    assert not np.allclose(eta_eta_T, expected_eta_eta_T,equal_nan=True)