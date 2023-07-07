# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:26:03 2023

@author: Maria Mihaescu
"""
import numpy as np
import matplotlib
from matplotlib import cm


from Binary_Alloys import ranges
from Binary_Alloys import xlog
from Binary_Alloys import interaction_parameter

from Binary_Alloys import enthalpy_scal
from Binary_Alloys import entropie_scal
from Binary_Alloys import free_energy_XB_eta

from Binary_Alloys import enthalpie_XB_T
from Binary_Alloys import T_entropie_XB_T
from Binary_Alloys import free_energy_XB_T

from Binary_Alloys import enthalpie_eta_T
from Binary_Alloys import T_entropie_eta_T
from Binary_Alloys import free_energy_eta_T

#Set the physical constants

k_B=1.38e-23  #Boltzmann's constant
N_A=6.02e+23  #Avogadro's number
R_gas=k_B*N_A #Gas constant

#####################################################################################################################
def test_ranges_valid():
    
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
    
###############################################################################################################


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
    
###############################################################################################################

def test_interaction_parameter_valid():
    
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

    assert np.isclose(result, expected_result)
    
###############################################################################################################


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
    
###############################################################################################################


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

###############################################################################################################


def test_free_energy_XB_eta_physical():
    """
    Test function for the free_energy_XB_eta in the physical range.
    
    The physical range is defines by abs(eta) strictly smaller than the composition (x) or (1-x)
    
    This test case verifies the calculation of free energy for specific input values.
    It checks if the calculated free energy matches the expected values.
    The length of the X_B and eta arrays have to be the same. 
    
    The composition values have to be positive and between 0 and 1 to have physical sense.
    and eta has to have values between -0.5 and 0.5.

    """
    # Input values
    T0 = 300.0
    omega = 10.0
    X_B = np.array([0.2, 0.4, 0.6])
    eta = np.array([-0.1, 0, 0.1])

    # Expected results
    expected_X_XB_eta = np.array([[0.2, 0.4, 0.6],
                                  [0.2, 0.4, 0.6],
                                  [0.2, 0.4, 0.6]])

    expected_eta_XB_eta = np.array([[-0.1, -0.1, -0.1],
                                    [0, 0, 0],
                                    [0.1 ,0.1, 0.1]])

    expected_G_XB_eta = np.array([[1.0234e24, 1.5050e24, 1.5050e24],
                                  [9.6320e23, 1.4448e24, 1.4448e24],
                                  [1.0234e24, 1.5050e24, 1.5050e24]])

    
    # Calculated results with the function
    X_XB_eta, eta_XB_eta, G_XB_eta = free_energy_XB_eta(T0, omega, X_B, eta)
    
    # Compare the calculated results with expected results
    
    assert np.allclose(X_XB_eta, expected_X_XB_eta)
    assert np.allclose(eta_XB_eta, expected_eta_XB_eta)
    assert np.allclose(G_XB_eta, expected_G_XB_eta)
    
    
def test_free_energy_XB_eta_unphysical():
    
    """
    Test function for the free_energy_XB_eta in the unphysical range.
    
    This test case verifies the calculation of free energy for specific input values.
    It checks if the calculated free energy matches the expected values.
    The length of the X_B and eta arrays have to be the same. 
    
    The composition values have to be positive and between 0 and 1 to have physical sense.
    and eta has to have values between -0.5 and 0.5.
    
    In the combinations where abs(eta) is greater than or equal to the composition (x) or (1-x),
    we expect np.nan values. This is the case for four combinations of eta and x, namely:
        - e=0.3 > x=0.2
        - e=0.5 > x=0.4
        - e=0.5 > (1-x)=1-0.6=0.4
        - e=0.5 > x=0.2
        
    This test looks at were the nan values are and if they are at the expected place for the free energy.
    """
    # Input values
    T0 = 300.0
    omega = 10.0
    X_B = np.array([0.2, 0.4, 0.6])
    eta = np.array([0.1, 0.3, 0.5])

    # Expected results
    expected_X_XB_eta = np.array([[0.2, 0.4, 0.6],
                                  [0.2, 0.4, 0.6],
                                  [0.2, 0.4, 0.6]])

    expected_eta_XB_eta = np.array([[0.1, 0.1, 0.1],
                                    [0.3, 0.3, 0.3],
                                    [0.5, 0.5, 0.5]])

    expected_nan_mask = np.array([[False, False, False],
                                  [True, False, False],
                                  [True, True, True]])
    
    # Calculated results with the function and looks where the np.nan values are
    
    X_XB_eta, eta_XB_eta, G_XB_eta = free_energy_XB_eta(T0, omega, X_B, eta)
    nan_indices = np.isnan(G_XB_eta)
    
    # Compare the calculated results with expected results
    
    assert np.allclose(X_XB_eta, expected_X_XB_eta)
    assert np.allclose(eta_XB_eta, expected_eta_XB_eta)
    assert np.allclose(nan_indices, expected_nan_mask)
    
###############################################################################################################


def test_enthalpie_XB_T_valid():
    """
    Test function for the T_enthalpie_XB_T function with valid input values.
    
    This test case verifies the calculation of enthalpy for specific input values.
    It checks if the calculated enthalpy matches the expected values.

    The input values include a composition array (XB) and a temperature array (T).
    the composition has to be between 0 and 1 to have physical sens.
    T0 is a temperature in kelvin and has to be positive  of a few 100 of K to have physical sens.
    
    This is the only case tested as those are requirements on the input values given to the user
    in the documentation. 

    """

    # Input values
    X_XB_T = np.array([[0.1, 0.2, 0.3],
                       [0.4, 0.5, 0.6],
                       [0.7, 0.8, 0.9]])
    T0 = 300.0
    omega = 10.0

    # Expected results
    expected_h = np.array([[5.4180e23, 9.6320e23, 1.2642e24],
                           [1.4448e24, 1.5050e24, 1.4448e24],
                           [1.2642e24, 9.6320e23, 5.4180e23]])

    # Calculated results with the function
    h = enthalpie_XB_T(X_XB_T, T0, omega)

    # Compare the calculated results with expected results
    assert np.allclose(h, expected_h)
    
###############################################################################################################


def test_T_entropie_XB_T_defined():
    
    """
    Test function for the T_entropie_XB_T function with valid input values.
    This test case verifies the calculation of temperature times entropy for valid input values.
    It checks if the calculated values match the expected results.

    The input values include a composition array (XB), a temperature array (T_XB_T),
    a reference temperature (T0), and an interaction parameter (omega).
    
    The composition has to be between 0 and 1 to have physical sense.
    The entropie is defined if the composition is strictly superior at 0 and strictly inferior 
    at 1.
    
    T_XB_T and T0 have to be positive, and omega can be any real value.

    """

    # Input values
    X_XB_T = np.array([[0.1, 0.2, 0.3],
                       [0.4, 0.5, 0.6],
                       [0.7, 0.8, 0.9]])
    
    T_XB_T = np.array([[200, 250, 300],
                       [350, 400, 450],
                       [500, 550, 600]])
    T0 = 300.0
    omega = 10.0

    # Expected results
    expected_ts = np.array([[540.13186195, 1039.28579345, 1522.44488273],
                            [1956.8891037, 2303.35580689, 2516.00027618],
                            [2537.40813788, 2286.42874558, 1620.39558585]])
    
    

    # Calculated results with the function
    ts = T_entropie_XB_T(X_XB_T, T_XB_T, T0, omega)

    # Compare the calculated results with expected results
    assert np.allclose(ts, expected_ts)
    
    
def test_T_entropie_XB_T_undefined():
    
    """
    Test function for the T_entropie_XB_T function with valid input values.
    This test case verifies the calculation of temperature times entropy for valid input values.
    It checks if the calculated values match the expected results.

    The input values include a composition array (XB), a temperature array (T_XB_T),
    a reference temperature (T0), and an interaction parameter (omega).
    
    We expect np.nan values for x=0 and x=1 because of the limits of the xlog function.
    And for negative compositions or composition bigger than 1 that don't have physical sense.
    
    The entropie is not defined in this limit.
    
    T_XB_T and T0 have to be positive, and omega can be any real value.

    """

    # Input values
    X_XB_T = np.array([[0, 0, 0],
                       [0.5, 0.1, -5],
                       [0.9, 1.2, 1]])
    
    T_XB_T = np.array([[200, 250, 300],
                       [350, 400, 450],
                       [500, 550, 600]])
    T0 = 300.0
    omega = 10.0

    # Expected results
    
    expected_nan_mask = np.array([[True, True, True],
                                  [False, False, True],
                                  [False, True, True]])

    # Calculated results with the function
    ts = T_entropie_XB_T(X_XB_T, T_XB_T, T0, omega)
    nan_indices = np.isnan(ts)

    # Compare the calculated results with expected results
    assert np.allclose(nan_indices, expected_nan_mask)
    
###############################################################################################################


def test_free_energy_XB_T_defined():
    
    """
    Test function for the free_energy_XB_T function with valid input values in the defined range.
    This test case verifies the calculation of free energy for valid input values.
    It checks if the calculated free energy matches the expected results.

    The input values include a reference temperature (T0), an interaction parameter (omega),
    a composition range (X_B), and a temperature range (T).
    The composition values have to be between 0 and 1 to have physical sense.
    The entropy is defined if the composition is strictly greater than 0 and strictly less than 1.
    T and T0 have to be positive, and omega can be any real value.
    """

    # Input values
    T0 = 300.0
    omega = 10.0
    X_B = np.array([0.2, 0.4, 0.6])
    T = np.array([300, 400, 500])

    # Expected results
    expected_X_XB_T = np.array([[0.2, 0.4, 0.6],
                                [0.2, 0.4, 0.6],
                                [0.2, 0.4, 0.6]])

    expected_T_XB_T = np.array([[300, 300, 300],
                                [400, 400, 400],
                                [500, 500, 500]])

    expected_G_XB_T = np.array([[9.6320e23, 1.4448e24, 1.4448e24],
                                [9.6320e23, 1.4448e24, 1.4448e24],
                                [9.6320e23, 1.4448e24, 1.4448e24]])

    # Calculated results with the function
    X_XB_T, T_XB_T, G_XB_T = free_energy_XB_T(T0, omega, X_B, T)

    # Compare the calculated results with expected results
    assert np.allclose(X_XB_T, expected_X_XB_T)
    assert np.allclose(T_XB_T, expected_T_XB_T)
    assert np.allclose(G_XB_T, expected_G_XB_T)



def test_free_energy_XB_T_undefined():
    
    """
    Test function for the free_energy_XB_T function with valid input values in the undefined range.
    This test case verifies the calculation of free energy for valid input values.
    It checks if the calculated free energy matches the expected results.

    The input values include a reference temperature (T0), an interaction parameter (omega),
    a composition range (X_B), and a temperature range (T).
    
    We expect np.nan values for X=0 and X=1 because of the limits of the xlog function.
    And for negative compositions or compositions greater than 1 that don't have physical sense.
    The entropy and free energy are not defined in these limits.
    
    T and T0 have to be positive, and omega can be any real value.
    """

    # Input values
    T0 = 300.0
    omega = 10.0
    X_B = np.array([0, 0.5, 1.2])
    T = np.array([300, 400, 500])

    # Expected results
    
    expected_X_XB_T = np.array([[0,  0.5, 1.2],
                                 [0,  0.5, 1.2],
                                 [0,  0.5, 1.2]])

    expected_T_XB_T = np.array([[300, 300, 300],
                                [400, 400, 400],
                                [500, 500, 500]])

    expected_nan_mask = np.array([[True, False, True],
                                  [True, False, True],
                                  [True, False, True]])

    # Calculated results with the function
    X_XB_T, T_XB_T, G_XB_T = free_energy_XB_T(T0, omega, X_B, T)
    nan_indices = np.isnan(G_XB_T)

    # Compare the calculated results with expected results
    assert np.allclose(X_XB_T, expected_X_XB_T)
    assert np.allclose(T_XB_T, expected_T_XB_T)
    assert np.allclose(nan_indices, expected_nan_mask)
    
###############################################################################################################


def test_enthalpie_eta_T_valid():
    """
    Test function for the enthalpie_eta_T function with valid input values.
    This test case verifies the calculation of enthalpy for valid input values.
    It checks if the calculated values match the expected results.

    The input values include a structural order parameter array (eta_eta_T),
    a reference temperature (T0), and an interaction parameter (omega).
    The structural order parameter values and temperature values have to be within the appropriate ranges.
    (eta between -0.5 and 0.5 and positive temperatures) to have physical sens.
    """

    # Input values
    T0 = 300.0
    omega = 10.0
    
    eta_eta_T = np.array([[-0.5, -0.3, -0.1],
                          [0, 0.2, 0.3],
                          [0.4, 0.5, 0]])

    # Expected results
    expected_h = np.array([[3.0100e24, 2.0468e24, 1.5652e24],
                           [1.5050e24, 1.7458e24, 2.0468e24],
                           [2.4682e24, 3.0100e24, 1.5050e24]])

    # Calculated results with the function
    h = enthalpie_eta_T(eta_eta_T, T0, omega)

    # Compare the calculated results with expected results
    assert np.allclose(h, expected_h)
    
###############################################################################################################


def test_T_entropie_eta_T_defined():
    
    """
    Test function for the T_entropie_eta_T function with valid input values.
    This test case verifies the calculation of temperature times entropy for valid input values in the defined range.
    It checks if the calculated values match the expected results.

    The input values include a structural order parameter array (eta_eta_T),
    a temperature array (T_XB_T), a reference temperature (T0), and an interaction parameter (omega).
    The structural order parameter values, temperature values, and reference temperature have to be within the appropriate ranges.
           - eta strictly between -0.5 and 0.5 (the entropie is not defined for eta=-0.5 or eta=0.5 or eta outside the range)
           - positive temperatures
    """

    # Input values
    T0 = 300.0
    omega = 10.0
    
    eta_eta_T = np.array([[-0.36, -0.1, -0.47],
                          [0, 0.16, 0.28],
                          [0.49, 0.48, -0.49]])
    
    T_XB_T = np.array([[200, 250, 300],
                       [350, 400, 450],
                       [500, 550, 600]])

    # Expected results
    
    expected_ts = np.array([[672.8549297, 1397.77793121, 335.81521091],
                            [2015.43633103, 2130.18653442, 1969.80326117],
                            [232.6191734, 447.95835562, 279.14300808]])

    # Calculated results with the function
    ts = T_entropie_eta_T(eta_eta_T, T_XB_T, T0, omega)

    # Compare the calculated results with expected results
    assert np.allclose(ts, expected_ts)

def test_T_entropie_eta_T_undefined():
    
    """
    Test function for the T_entropie_eta_T function with valid input values.
    This test case verifies the calculation of temperature times entropy for valid input values in the undefined range.
    It checks if the calculated values match the expected results.

    The input values include a structural order parameter array (eta_eta_T),
    a temperature array (T_XB_T), a reference temperature (T0), and an interaction parameter (omega).
    The structural order parameter values, temperature values, and reference temperature have to be within the appropriate ranges.
           - the entropie is not defined for eta=-0.5 or eta=0.5 or eta outside the range
           -we expect np.nan values of the entropie for those values of eta
           - positive temperatures
    """

    # Input values
    T0 = 300.0
    omega = 10.0
    
    eta_eta_T = np.array([[-0.5, -0.1, -0.47],
                          [0, 0.16, 0.6],
                          [-0.9, 0.5, -0.49]])
    
    T_XB_T = np.array([[200, 250, 300],
                       [350, 400, 450],
                       [500, 550, 600]])

    # Expected results
    
    expected_nan_mask = np.array([[True, False, False],
                                  [False, False, True],
                                  [True, True, False]])

    # Calculated results with the function
    ts = T_entropie_eta_T(eta_eta_T, T_XB_T, T0, omega)
    nan_indices = np.isnan(ts)

    # Compare the calculated results with expected results
    assert np.allclose(nan_indices, expected_nan_mask)
    
###############################################################################################################

def test_free_energy_eta_T_defined():
    
    """
    Test function for the free_energy_eta_T function with valid input values in the defined range.
    This test case verifies the calculation of free energy for valid input values.
    It checks if the calculated free energy matches the expected results.

    The input values include a reference temperature (T0), an interaction parameter (omega),
    a composition range (X_B), a structural order parameter range (eta), and a temperature range (T).
    The composition, structural order parameter, and temperature values have to be within the appropriate ranges.
        - X_B strictly between 0 and 1
        - eta strictly between -0.5 and 0.5
        - positive temperatures

    """

    # Input values
    T0 = 300.0
    omega = 10.0
    X_B = np.array([0.2, 0.4, 0.6])
    eta = np.array([-0.3, 0.0, 0.3])
    T = np.array([300, 400, 500])

    # Expected results
    expected_eta_eta_T = np.array([[-0.3, 0, 0.3],
                                   [-0.3, 0.0, 0.3],
                                   [-0.3, 0, 0.3]])

    expected_T_eta_T = np.array([[300, 300, 300],
                                 [400, 400, 400],
                                 [500, 500, 500]])

    expected_G_eta_T = np.array([[2.0468e24, 1.5050e24, 2.0468e24],
                                 [2.0468e24, 1.5050e24, 2.0468e24],
                                 [2.0468e24, 1.5050e24, 2.0468e24]])

    # Calculated results with the function
    eta_eta_T, T_eta_T, G_eta_T = free_energy_eta_T(T0, omega, X_B, eta, T)

    # Compare the calculated results with expected results
    assert np.allclose(eta_eta_T, expected_eta_eta_T)
    assert np.allclose(T_eta_T, expected_T_eta_T)
    assert np.allclose(G_eta_T, expected_G_eta_T)

def test_free_energy_eta_T_undefined():
    
    """
    Test function for the free_energy_eta_T function with valid input values in the undefined range.
    This test case verifies the calculation of free energy for valid input values.
    It checks if the calculated free energy matches the expected results.

    The input values include a reference temperature (T0), an interaction parameter (omega),
    a composition range (X_B), a structural order parameter range (eta), and a temperature range (T).
    The composition, structural order parameter, and temperature values have to be within the appropriate ranges.
        - X_B strictly between 0 and 1
        - eta strictly between -0.5 and 0.5
        - positive temperatures
    The entropie is not defined outside those ranges and we expect np.nan values for the free
    energy outside of the ranges and at the limits x=0, x=1, eta=-0.5 and eta=0.5.
    

    """

    # Input values
    T0 = 300.0
    omega = 10.0
    X_B = np.array([0.2, 0.4, 1])
    eta = np.array([-0.5, 0.0, 0.5])
    T = np.array([300, 400, 500])

    # Expected results
    expected_eta_eta_T = np.array([[-0.5, 0, 0.5],
                                   [-0.5, 0.0, 0.5],
                                   [-0.5, 0, 0.5]])

    expected_T_eta_T = np.array([[300, 300, 300],
                                 [400, 400, 400],
                                 [500, 500, 500]])
    
    expected_nan_mask = np.array([[True, False, True],
                                  [True, False, True],
                                  [True, False, True]])

    # Calculated results with the function
    eta_eta_T, T_eta_T, G_eta_T = free_energy_eta_T(T0, omega, X_B, eta, T)
    nan_indices = np.isnan(G_eta_T)

    # Compare the calculated results with expected results
    assert np.allclose(eta_eta_T, expected_eta_eta_T)
    assert np.allclose(T_eta_T, expected_T_eta_T)
    assert np.allclose(nan_indices, expected_nan_mask)
    

###############################################################################################################
