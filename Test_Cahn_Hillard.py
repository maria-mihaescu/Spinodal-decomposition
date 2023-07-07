# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:41:21 2023

@author: Maria Mihaescu
"""
import numpy as np

#Defining the physical constants
R = 8.314 # gas constant

from Cahn_Hillard import atom_interac_cst
from Cahn_Hillard import diffusion_coeff


#####################################################################################################################

def test_atom_interac_cst_positive():
    """
    Test function for the atom_interac_cst function.
    This test case verifies the calculation of the atomic interaction constant for specific input values.
    It checks if the calculated atomic interaction constant matches the expected value.

    The input value is a temperature (T) in Kelvin that has to be positive or zero.

    """

    # Input value
    T = 300.0

    # Expected result
    expected_La = 17300.0

    # Calculated result with the function
    La = atom_interac_cst(T)
    # Compare the calculated result with expected result
    assert np.isclose(La, expected_La)

def test_atom_interac_cst_zero():
    """
    Test function for the atom_interac_cst function.
    This test case verifies the calculation of the atomic interaction constant for specific input values.
    It checks if the calculated atomic interaction constant matches the expected value.

    The input value is a temperature (T) for the special case T=0.

    """

    # Input value
    T = 0

    # Expected result
    expected_La = 20000.0

    # Calculated result with the function
    La = atom_interac_cst(T)
    # Compare the calculated result with expected result
    assert np.isclose(La, expected_La)
    
#####################################################################################################################

def test_diffusion_coeff_positive():
    """
    Test function for the diffusion_coeff function with positive coefficient, energy, and temperature.
    The temperature has to be strictly positive it cannot be 0.
    
    This test case verifies the calculation of the diffusion coefficient for positive input values.
    It checks if the calculated diffusion coefficient matches the expected result.


    """

    # Input values
    coef = 1.0
    E = 5000.0
    T = 300.0

    # Expected result
    expected_diff = 0.1347073286516485

    # Calculated result with the function
    diff = diffusion_coeff(coef, E, T)


    # Compare the calculated result with the expected result
    assert np.allclose(diff, expected_diff)


def test_diffusion_coeff_zero_coef():
    """
    Test function for the diffusion_coeff function with zero coefficient.
    The temperature has to be strictly positive it cannot be 0.
    
    This test case verifies the calculation of the diffusion coefficient for zero coefficient.
    It checks if the calculated diffusion coefficient matches the expected result which is 0.

    """

    # Input values
    coef = 0.0
    E = 5000.0
    T = 300.0

    # Expected result
    expected_diff = 0.0  

    # Calculated result with the function
    diff = diffusion_coeff(coef, E, T)

    # Compare the calculated result with the expected result
    assert np.allclose(diff, expected_diff)
    
def test_diffusion_coeff_zero_energy():
    
    """
    Test function for the diffusion_coeff function with zero coefficient.
    The temperature has to be strictly positive it cannot be 0.
    
    This test case verifies the calculation of the diffusion coefficient for an energy equal to 0.
    
    It checks if the calculated diffusion coefficient matches the expected result which is equal to 
    the coefficient of diffusion.

    """

    # Input values
    coef = 2e-05
    E = 0
    T = 300.0

    # Expected result
    expected_diff = 2e-05

    # Calculated result with the function
    diff = diffusion_coeff(coef, E, T)

    # Compare the calculated result with the expected result
    assert np.allclose(diff, expected_diff)




#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

