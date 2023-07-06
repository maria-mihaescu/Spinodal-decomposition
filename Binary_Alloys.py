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

def test_ranges():
    
    """
    This test function tests if the ranges function takes well the minimum, maximum and step value for each
    quantity and gives back three arrays that have the expected range values.

    The starting data have the caracteristics that the min value is smaller that the max value and that the 
    step value is smaller that the two values. This is the only case tested as this is a requirement given 
    to the user in the documentation. 
    
    The expected values are that the first value is equal to the min value of the range, the max value of the range
    is equal to the maximal set value minus one step, and the values inbetween are separate on one step value. 
    
    """
    #Starting data for the composition range
    X_B_min = 0.0
    X_B_max = 1.0
    X_B_step = 0.2
    
    #starting data for the temperature range
    T_min = 300.0
    T_max = 500.0
    T_step = 100.0
    
    #starting data for the structural order parameter range
    eta_min = 0.1
    eta_max = 0.5
    eta_step = 0.2

    #expected output ranges for the three parameters
    
    expected_X_B = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    expected_T = np.array([300.0, 400.0])
    expected_eta = np.array([0.1, 0.3])

    #calcultated ranges with the function 
    X_B, T, eta = ranges(X_B_min, X_B_max, X_B_step, T_min, T_max, T_step, eta_min, eta_max, eta_step)


    assert np.allclose(X_B, expected_X_B)
    assert np.allclose(T, expected_T)
    assert np.allclose(eta, expected_eta)


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


def test_xlog_positive():
    """
    Test function for the x_log function if we enter an array of only positive numbers smaller than 1.
    
    The starting data is three positive floats smaller than 1. We expect that the 
    xlog function gives negative values. 

    """
    #Starting testing data
    X1 = np.array([0.2, 0.5, 0.8])
    
    #expected results
    expected_result1 = np.array([-0.32188758, -0.34657359, -0.17851484])
    
    #results calculated with the function xlog
    result1 = xlog(X1)

    assert np.allclose(result1, expected_result1)
    

    
def test_xlog_negative():
    """
    Test function for the x_log function if we enter an array of only negative numbers.
    
    The starting data is four negative floats and integers. We expect that the 
    xlog function gives np.nan values for all of them as the logarithm of a 
    negative number is undefined. 

    """
    #starting testing data
    X2 = np.array([-1.2, -4., -0.03, -500])
    
    #expected results
    expected_result2 = np.array([np.nan, np.nan, np.nan, np.nan])
    
    #calculated results with the function 
    result2 = xlog(X2)
    
    assert np.allclose(result2, expected_result2, equal_nan=True)
    
def test_xlog_empty():
    """
    Test function for the x_log function if we enter an empty array.
    We expect that the xlog function gives an empty array as an output.

    """
    #starting testing data
    X3 = np.array([])
    
    #expected results
    expected_result3 = np.array([])
    
    #calculated results with the function 
    result3 = xlog(X3)
    
    assert np.array_equal(result3, expected_result3)
    
def test_xlog_zero():
    """
    Test function for the x_log function if we enter an array with only one zero value.
    We expect that the xlog function gives an array with an np.nan as the logarithm to 0 in undefined.

    """
    #starting testing data
    X4= np.array([0])
    
    #expected results
    expected_result4 = np.array([np.nan])
    
    #calculated results with the function 
    result4 = xlog(X4)
    
    assert np.array_equal(result4, expected_result4,equal_nan=True)
    
def test_xlog_nan():
    """
    Test function for the x_log function if we enter an array with np.nan values.
    We expect that the xlog function gives an array with the same number of np.nan values.

    """
    #starting testing data
    X5= np.array([np.nan, np.nan, np.nan, np.nan])
    
    #expected results
    expected_result5 = np.array([np.nan, np.nan, np.nan, np.nan])
    
    #calculated results with the function 
    result5 = xlog(X5)
    
    assert np.array_equal(result5, expected_result5,equal_nan=True)
    
    
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

def test_interaction_parameter():
    
    """
    test the calculation of the interaction parameter.
    
    The input data is that Z is a positive integer, diff_eV a positive float 
    and that the expected result is the multiplication of the two with the conversion
    from eV to Joules.
    
    This is the only case tested as those are requirements given to the user in the documentation. 
    We expect a negative value as the output as we multiply three positive values between themselves and -1. 
    """
    #Starting data
    Z = 10
    diff_eV = 0.5
    
    #expected result
    expected_result = -8e-18

    #result calculated with the function
    result = interaction_parameter(Z, diff_eV)

    assert np.isclose(result, expected_result), "interaction_parameter test failed"


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



def test_enthalpy_scal_physical_region():
    
    """
    This test case verifies the calculation of the enthalpy when the absolute value
    of the order parameter (eta) is lower than the composition (x) and (1-x), as stated in the model.
    It checks if the calculated enthalpy matches the expected value in the physical region.

    """
    #input values
    eta = 0.2
    composition = 0.4
    omega = 10.0
    T0 = 300.0

    #expected output
    expected_h = 1.6856e24

    #calculated output 
    h = enthalpy_scal(eta, composition, omega, T0)

    assert np.isclose(h, expected_h)


def test_enthalpy_scal_unphysical_region():
    
    """
    This test case verifies the handling of the unphysical region in the (X_B, eta) mesh.
    When the absolute value of the order parameter (eta) is greater than or equal to the composition (x)
    or (1-x), the enthalpy should be set to "Not a Number" (NaN) to avoid plotting non-physical values.
    It checks if the calculated enthalpy matches the expected NaN value.
    """
    
    #input data
    eta = 0.5
    composition = 0.8
    omega = 5.0
    T0 = 500.0

    #expected output
    expected_h = np.nan

    #calculated value
    h = enthalpy_scal(eta, composition, omega, T0)

    assert np.isclose(h, expected_h,equal_nan=True)

def test_enthalpy_scal_equal():
    """
    This test case verifies the calculation of the enthalpy when the absolute value
    of the order parameter (eta) is equal to the composition (x), it is a limit case
    as eta is smaler than (1-x) but not smaler than x.
    We expect the output value to be not a number as it is in the unphysical region

    """

    #Input data
    eta = 0.1
    composition = 0.1
    omega = 2.0
    T0 = 400.0

    #expected output
    expected_h = np.nan

    #calculated output
    h = enthalpy_scal(eta, composition, omega, T0)

    assert np.isclose(h, expected_h,equal_nan=True)


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


def test_entropie_scal_physical():
    """
    This test case verifies the calculation of the enthalpy when the absolute value
    of the order parameter (eta) is lower than the composition (x) and (1-x), as stated in the model.
    It checks if the calculated enthalpy matches the expected value in the physical region.

    """
    # Input values
    eta = 0.2
    composition = 0.4
    omega = 10.0
    T0 = 300.0

    # Expected output
    expected_s = 4.874127449315975

    # Calculated output
    s = entropie_scal(eta, composition, omega, T0)

    assert np.isclose(s, expected_s)


def test_entropie_scal_unphysical():
    
    """
    Test function for the entropie_scal() function with unphysical inputs.
    It checks if the calculated entropy is set to NaN as expected.

    Test Case:
    - Unphysical input values where abs(eta) >= x or abs(eta) >= (1 - x).

    """
    # Input values
    eta = 0.6
    composition = 0.8
    omega = 5.0
    T0 = 400.0

    # Expected output (NaN for unphysical input)
    expected_s = np.nan

    # Calculated output
    s = entropie_scal(eta, composition, omega, T0)

    assert np.isclose(s,expected_s,equal_nan=True)

def test_entropie_scal_equal():
    """
    This test case verifies the calculation of the entropie when the absolute value
    of the order parameter (eta) is equal to the composition (x), it is a limit case
    as eta is smaler than (1-x) but not smaler than x.
    We expect the output value to be not a number as it is in the unphysical region

    """

    #Input data
    eta = 0.1
    composition = 0.1
    omega = 2.0
    T0 = 400.0

    #expected output
    expected_s = np.nan

    #calculated output
    s = entropie_scal(eta, composition, omega, T0)

    assert np.isclose(s, expected_s,equal_nan=True)

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




def test_free_energy_XB_eta():
    
    """
    Test function for the definition of the free energy in the (XB,eta) space

    Returns
    -------
    None.

    """
    
    T0 = 300  # example value for T0
    omega = 1.5  # example value for omega
    X_B=np.array(0,1,0.01)
    eta=np.array(-0.5,0.5,0.01)
    
    X_XB_eta, eta_XB_eta, G_XB_eta = free_energy_XB_eta(T0,omega,X_B,eta)

    # Check if any NaN values exist in the arrays
    assert np.isnan(G_XB_eta).any()  # Expecting the presence of NaN values

    
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
    

    # Call the function
    X_XB_T, T_XB_T, G_XB_T = free_energy_XB_T(T0, omega,X_B,T)
    
    # Perform assertions
    assert isinstance(X_XB_T, np.ndarray)
    assert isinstance(T_XB_T, np.ndarray)
    assert isinstance(G_XB_T, np.ndarray)

    
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
    X_B=np.array(0,1,0.1)
    eta=np.arrange(-0.5,0.5,0.01)
    T=np.arrange(50,1000,50)
    # Call the function
    eta_eta_T, T_eta_T, G_eta_T = free_energy_eta_T(T0, omega, X_B,eta, T)
    print(eta_eta_T)
    # Perform assertions
    assert isinstance(eta_eta_T, np.ndarray)
    assert isinstance(T_eta_T, np.ndarray)
    assert isinstance(G_eta_T, np.ndarray)

    # Check shapes of the output arrays
    assert eta_eta_T.shape == T_eta_T.shape
    assert eta_eta_T.shape == G_eta_T.shape

    #assertion for eta_eta_T calculation
    X_XB_T,T_XB_T,G_XB_T = free_energy_XB_T(T0,omega,X_B,T)
    expected_eta_eta_T = enthalpie_eta_T(eta_eta_T, T0, omega) - T_entropie_eta_T(eta_eta_T, T_XB_T, T0, omega)
    
    assert not np.allclose(eta_eta_T, expected_eta_eta_T,equal_nan=True)