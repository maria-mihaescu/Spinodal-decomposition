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
from Cahn_Hillard import time_increment

from Cahn_Hillard import add_fluctuation
from Cahn_Hillard import func_laplacian

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

def test_time_increment_positive():
    """
    Test function for the time_increment function with positive inputs.
    This test case verifies the calculation of the time increment for positive input values.
    It checks if the calculated time increment matches the expected result.
    
    This is the only physically valid case as dx has to be positive and different to zero,
    and the diffusion coefficient has to be positive and defferent than zero.

    """

    # Input values
    dx = 0.1
    Diff_A = 1.0

    # Expected result
    expected_dt = 0.001  

    # Calculated result with the function
    dt = time_increment(dx, Diff_A)
    
    # Compare the calculated result with the expected result
    assert np.allclose(dt, expected_dt)


#####################################################################################################################

def test_add_fluctuation():
    """
    Test function for the add_fluctuation function.
    
    This test case verifies the addition of random fluctuations to the composition matrix.
    It checks if the resulting composition matrix falls within the expected range and if the seed produces reproducible results.

    """

    # Input values
    Nx = 10
    Ny = 10
    c0 = 0.5
    seed = 123  # Set a specific seed for reproducibility

    # Expected range
    expected_min = c0
    expected_max = c0 + 0.01  # Fluctuation range specified in the function

    # Calculated results with the function
    c1 = add_fluctuation(Nx, Ny, c0, seed=seed)
    c2 = add_fluctuation(Nx, Ny, c0, seed=seed)

    # Check if the composition matrices fall within the expected range
    assert np.all(c1 >= expected_min) and np.all(c1 <= expected_max)
    assert np.all(c2 >= expected_min) and np.all(c2 <= expected_max)

    # Check if the results with the same seed are reproducible
    assert np.array_equal(c1, c2)



#####################################################################################################################

def test_func_laplacian():
    
    """
    Test function for the func_laplacian function.
    This test case verifies the calculation of the gradient term for a given quantity at a matrix point.
    It checks if the calculated gradient term matches the expected result.

    The input values include the quantity at the center point and its surrounding points (left, right, up, down),
    as well as the grid spacing in the x and y directions (dx, dy).
    
    All those quantities have to be positive, and the grid spacing cannot be equal to 0.

    """

    # Input values
    center = 2.0
    left = 1.0
    right = 6
    up = 4.0
    down = 0.5
    dx = 0.1
    dy = 0.2

    # Expected result
    expected_lap = 312.5
    
    # Calculated result with the function
    lap = func_laplacian(center, left, right, up, down, dx, dy)

    # Compare the calculated result with expected result
    assert np.isclose(lap, expected_lap)


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

