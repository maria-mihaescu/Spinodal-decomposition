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
from Cahn_Hillard import chemical_free_energy_density

from Cahn_Hillard import boundary_conditions
from Cahn_Hillard import composition_nearest_neighbours

from Cahn_Hillard import diffusion_potential_chemical


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

def test_func_laplacian_positive():
    
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

def test_chemical_free_energy_density_positive():
    """
    Test function for the chemical_free_energy_density function.
    This test case verifies the calculation of the chemical potential at a specific point with concentration c.
    It checks if the calculated chemical free energy density matches the expected result.

    The input values include the 
        -composition of B atom (c), that has to be strictly between 0 and 1
        -the temperature (T), that has to be strictly positive
        -the atomic interaction constant (La) That has to be positive
        
    This is the only physically valid inputs, thus it is the only case tested.

    """

    # Input values
    c = 0.3
    T = 300.0
    La = 20000.0

    # Expected result
    expected_chem_pot = 2676.3822

    # Calculated result with the function
    chem_pot = chemical_free_energy_density(c, T, La)

    # Compare the calculated result with expected result
    assert np.isclose(chem_pot, expected_chem_pot)
    
#####################################################################################################################

def test_boundary_conditions_right():
    """
    Test function to verify the correctness of the boundary_conditions function for cells on the right side of the grid.
    This test case ensures that when the central cell is at the rightmost edge of the grid, the neighboring cell on
    the right side wraps around to the opposite edge.

    The input values include the coordinate positions (x, y) of the central cell, and the size of the computational grid (Nx * Ny).

    The expected output values are the coordinates of the central cell (x, y) and its neighboring cells (x_plus, x_min, y_plus, y_min).

    The calculated output values are obtained by calling the boundary_conditions function with the input values.

    The function compares the calculated output values with the expected output values using the assert statement.
    """

    # Test case
    x = 6
    y = 3
    Nx = 6
    Ny = 6

    # Expected results
    expected_output = (6, 3, 1, 5, 4, 2)

    # Calculated results with the function
    calculated_output = boundary_conditions(x, y, Nx, Ny)

    # Compare the calculated results with the expected results
    assert calculated_output == expected_output


def test_boundary_conditions_left():
    """
    Test function to verify the correctness of the boundary_conditions function for cells on the left side of the grid.
    This test case ensures that when the central cell is at the leftmost edge of the grid, the neighboring cell on
    the left side wraps around to the opposite edge.

    The input values include the coordinate positions (x, y) of the central cell, and the size of the computational grid (Nx * Ny).

    The expected output values are the coordinates of the central cell (x, y) and its neighboring cells (x_plus, x_min, y_plus, y_min).

    The calculated output values are obtained by calling the boundary_conditions function with the input values.

    The function compares the calculated output values with the expected output values using the assert statement.
    """

    # Test case
    x = 0
    y = 4
    Nx = 7
    Ny = 7

    # Expected results
    expected_output = (0, 4, 1, 6, 5, 3)

    # Calculated results with the function
    calculated_output = boundary_conditions(x, y, Nx, Ny)

    # Compare the calculated results with the expected results
    assert calculated_output == expected_output


def test_boundary_conditions_top():
    """
    Test function to verify the correctness of the boundary_conditions function for cells on the top side of the grid.
    This test case ensures that when the central cell is at the top edge of the grid, the neighboring cell above wraps
    around to the opposite edge.

    The input values include the coordinate positions (x, y) of the central cell, and the size of the computational grid (Nx * Ny).

    The expected output values are the coordinates of the central cell (x, y) and its neighboring cells (x_plus, x_min, y_plus, y_min).

    The calculated output values are obtained by calling the boundary_conditions function with the input values.

    The function compares the calculated output values with the expected output values using the assert statement.
    """

    # Test case
    x = 3
    y = 8
    Nx = 8
    Ny = 8

    # Expected results
    expected_output = (3, 8, 4, 2, 1, 7)

    # Calculated results with the function
    calculated_output = boundary_conditions(x, y, Nx, Ny)

    # Compare the calculated results with the expected results
    assert calculated_output == expected_output


def test_boundary_conditions_bottom():
    """
    Test function to verify the correctness of the boundary_conditions function for cells on the bottom side of the grid.
    This test case ensures that when the central cell is at the bottom edge of the grid, the neighboring cell below wraps
    around to the opposite edge.

    The input values include the coordinate positions (x, y) of the central cell, and the size of the computational grid (Nx * Ny).

    The expected output values are the coordinates of the central cell (x, y) and its neighboring cells (x_plus, x_min, y_plus, y_min).

    The calculated output values are obtained by calling the boundary_conditions function with the input values.

    The function compares the calculated output values with the expected output values using the assert statement.
    """

    # Test case
    x = 5
    y = 1
    Nx = 6
    Ny = 6

    # Expected results
    expected_output = (5, 1, 0, 4, 2, 0)

    # Calculated results with the function
    calculated_output = boundary_conditions(x, y, Nx, Ny)

    # Compare the calculated results with the expected results
    assert calculated_output == expected_output


#####################################################################################################################

def test_composition_nearest_neighbours_left_border():
    """
    Test function to verify the correctness of the composition_nearest_neighbours function for the left border.
    This test case checks the composition values of the nearest neighbors of a matrix, specifically for the left border.

    The composition matrix (c) has a size of 3x3.
    The coordinates (x, y) represent a cell on the left border, where x = 0.

    The expected composition values of the nearest neighbors are:
    c_center = 0.3, c_left = 0.9 (same cell), c_right = 0.6, c_up = 0.4, c_down = 0.2

    The calculated composition values are obtained by calling the composition_nearest_neighbours function with the input values.

    The function compares the calculated composition values with the expected composition values using the assert statement.
    """

    # Input values
    c = np.array([[0.2, 0.3, 0.4],
                  [0.5, 0.6, 0.7],
                  [0.8, 0.9, 1.0]])
    x = 0
    y = 1
    Nx = 3
    Ny = 3
    
    #expected result 
    expected_output = (0.3, 0.9, 0.6, 0.4, 0.2)
    
    #calculated result
    calculated_output = composition_nearest_neighbours(c, x, y, Nx, Ny)
    
    assert  calculated_output == expected_output


def test_composition_nearest_neighbours_right_border():
    """
    Test function to verify the correctness of the composition_nearest_neighbours function for the right border.
    This test case checks the composition values of the nearest neighbors of a matrix, specifically for the right border.

    The composition matrix (c) has a size of 4x4.
    The coordinates (x, y) represent a cell on the right border, where x = 3.

    The expected composition values of the nearest neighbors are:
    c_center = 0.4, c_left = 0.9, c_right = 0.3 (same cell), c_up = 0.6, c_down = 0.3 (same cell)

    The calculated composition values are obtained by calling the composition_nearest_neighbours function with the input values.

    The function compares the calculated composition values with the expected composition values using the assert statement.
    """

    # Test case
    c = np.array([[0.1, 0.2, 0.3, 0.5],
                  [0.4, 0.5, 0.6, 0.9],
                  [0.7, 0.8, 0.9, 1.0],
                  [0.2, 0.3, 0.4, 0.6]])
    x = 3
    y = 2
    Nx = 4
    Ny = 4
    
    #expected result 
    expected_output = (0.4, 0.9, 0.3, 0.6, 0.3)
    
    #calculated result
    calculated_output = composition_nearest_neighbours(c, x, y, Nx, Ny)
    
    assert  calculated_output == expected_output

def test_composition_nearest_neighbours_top_border():
    """
    Test function to verify the correctness of the composition_nearest_neighbours function for the top border.
    This test case checks the composition values of the nearest neighbors of a matrix, specifically for the top border.

    The composition matrix (c) has a size of 5x5.
    The coordinates (x, y) represent a cell on the top border, where y = 4.

    The expected composition values of the nearest neighbors are:
    c_center = 0.2, c_left = 0.3, c_right = 0.7, c_up = 0.4 (same cell), c_down = 0.4

    The calculated composition values are obtained by calling the composition_nearest_neighbours function with the input values.

    The function compares the calculated composition values with the expected composition values using the assert statement.
    """

    # Test case
    c = np.array([[0.3, 0.2, 0.1, 0.8, 0.5],
                  [0.5, 0.6, 0.7, 0.6, 0.3],
                  [0.4, 0.5, 0.6, 0.4, 0.2],
                  [0.6, 0.7, 0.8, 0.9, 0.7],
                  [0.8, 0.9, 1.0, 0.7, 0.6]])
    x = 2
    y = 4
    Nx = 5
    Ny = 5
    
    #expected result
    expected_output = (0.2, 0.3, 0.7, 0.4, 0.4)
    
    
    #calculated result
    calculated_output = composition_nearest_neighbours(c, x, y, Nx, Ny)
    
    assert  calculated_output == expected_output


def test_composition_nearest_neighbours_bottom_border():
    """
    Test function to verify the correctness of the composition_nearest_neighbours function for the bottom border.
    This test case checks the composition values of the nearest neighbors of a matrix, specifically for the bottom border.

    The composition matrix (c) has a size of 5x5.
    The coordinates (x, y) represent a cell on the bottom border, where y = 0.

    The expected composition values of the nearest neighbors are:
    c_center = 0.2, c_left = 0.4, c_right = 0.5, c_up = 0.3 (same cell), c_down = 0.4

    The calculated composition values are obtained by calling the composition_nearest_neighbours function with the input values.

    The function compares the calculated composition values with the expected composition values using the assert statement.
    """

    # Test case
    c = np.array([[0.5, 0.9, 0.7, 0.4, 0.3],
                  [0.6, 0.7, 0.8, 0.5, 0.2],
                  [0.4, 0.5, 0.6, 0.3, 0.1],
                  [0.2, 0.3, 0.4, 0.5, 0.4],
                  [0.5, 0.6, 0.7, 0.5, 0.3]])
    x = 3
    y = 0
    Nx = 5
    Ny = 5
    
    #expected result 
    expected_output = (0.2, 0.4, 0.5, 0.3, 0.4)
    
    #calculated result
    calculated_output = composition_nearest_neighbours(c, x, y, Nx, Ny)
    
    assert  calculated_output == expected_output

#####################################################################################################################


def test_diffusion_potential_chemical_cc_0_5():
    """
    Test function to verify the correctness of the diffusion_potential_chemical function when the composition (cc) is 0.5.
    The composition has to be strictly between 0 and 1 to make physical sens.

    For cc=0.5 there is the same amount of A and B atoms. 
    We expect the ouput to be 0.

    The function compares the calculated output with the expected output using the assert statement.
    """

    # Test case
    cc = 0.5
    T = 400.0
    La = 150000.0
    
    #Expected result
    expected_output = 0.0

    # Calculated output with the function
    calculated_output = diffusion_potential_chemical(cc, T, La)

    # Compare the calculated output with the expected output
    assert calculated_output == expected_output


def test_diffusion_potential_chemical_cc_0_8():
    """
    Test function to verify the correctness of the diffusion_potential_chemical function when the composition (cc) is 0.8.
    The composition has to be strictly between 0 and 1 to make physical sens.

    For cc=0.8 there are more B atoms than A atoms.

    The function compares the calculated output with the expected output using the assert statement.
    """

    # Test case
    cc = 0.8
    T = 400.0
    La = 150000.0
    
    #Expected result
    expected_output = -85389.73947265971

    # Calculated output with the function
    calculated_output = diffusion_potential_chemical(cc, T, La)

    # Compare the calculated output with the expected output
    assert calculated_output == expected_output

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

#####################################################################################################################

