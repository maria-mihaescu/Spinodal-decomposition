# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:46:02 2023

@author: Maria Mihaescu
"""

import numpy as np
import matplotlib


#Defining the physical constants
R = 8.314 # gas constant

def atom_interac_cst(T):
    """
    Function to calculate the atomic interaction constant between atom A and B in  [J/mol]

    Parameters
    ----------
    T : TYPE float
        DESCRIPTION.Temperature in K

    Returns
    -------
    La : TYPE float
         DESCRIPTION atomic interaction constant between atom A and B in  [J/mol]

    """    
    La = 20000.-9.*T # Atom interaction constant [J/mol]
    return La


def diffusion_coeff(coef,
                    E,
                    T):
    """
    Exponential function to calculate the diffusion coefficient in [m2/s]

    Parameters
    ----------
    coef : TYPE float
        DESCRIPTION. Coefficient used for the calculation of the diffusion coefficient
    E : TYPE float
        DESCRIPTION.Energy in J
    T : TYPE float
        DESCRIPTION. Temperature in K 

    Returns
    -------
    Diff : TYPE float 
        DESCRIPTION. diffusion coefficient in [m2/s]

    """
    Diff=coef*np.exp(-E/(R*T))
    return Diff


def time_increment(dx,
                   Diff_A):
    """
    Function to calculate the time increment in s between each update of the composition

    Parameters
    ----------
    dx : TYPE float
        DESCRIPTION. spacing of the computational grid in the x direction [m]
    Diff_A : TYPE float 
        DESCRIPTION.diffusion coefficient of atom A in [m2/s]

    Returns
    -------
    dt : TYPE float 
        DESCRIPTION.time increment [s]

    """
    conversion_factor=0.1 #converts dt in seconds
    dt = (dx*dx/Diff_A)*conversion_factor
    return dt



def add_fluctuation(Nx, 
                    Ny, 
                    c0,
                    seed=None): 
    """
    function that adds random fluctuation of the composition in a computational grid of size
    Nx*Ny, starting from an initial composition of B atoms of c0 in [atomic fraction]

    Parameters
    ----------
    Nx : TYPE int
        DESCRIPTION. number of computational grids along the x direction
    Ny : TYPE int
        DESCRIPTION. number of computational grids along the y direction
    c0 : TYPE float
        DESCRIPTION. average composition of B atom [atomic fraction]
    seed : TYPE int or None, optional
        DESCRIPTION. random seed for reproducibility


    Returns
    -------
    c : TYPE list 
        DESCRIPTION. Composition matrix where c[x,y] is the composition at point of coordinate (x,y)

    """
    if seed is not None:
        np.random.seed(seed)
        
    conversion_factor=0.01 #gets the composition in atomic fraction
    
    c = c0 + np.random.rand(Nx, Ny)*conversion_factor
    
    return c

  
def func_laplacian(center,
                   left,
                   right,
                   up,
                   down,
                   dx,
                   dy):
    """
    Calculates the gradient term of a quantity at a given matrix point
    (composition or diffusion potential) given the variation of the quantity 
    within its nearest neighbours in the matrix of size Nx*Ny.

    Parameters
    ----------
    center : TYPE float
        DESCRIPTION. quantity at the point (x,y)
    left : TYPE float
        DESCRIPTION. quantity at the point (x-1,y)
    right : TYPE float
        DESCRIPTION. quantity at the point (x+1,y)
    up : TYPE float
        DESCRIPTION. quantity at the point (x,y+1)
    down : TYPE float
        DESCRIPTION. quantity at the point (x,y-1)
    dx : TYPE float
        DESCRIPTION. spacing of the computational grid in the x direction [m]
    dy : TYPE float
        DESCRIPTION. spacing of the computational grid in the y direction [m]

    Returns
    -------
    lap : TYPE float
        DESCRIPTION. gradient term of the quantity at a matrix point

    """
    lap=(right-2*center+left)/dx/dx + (up -2*center+down) /dy/dy
    return lap


def chemical_free_energy_density(c,
                       T,
                       La):
    """
    Function to calculate the chemical potential  in a specific point given a composition of B atom, a temperature and 
    and atomic interaction constante

    Parameters
    ----------
    c : TYPE float
        DESCRIPTION. composition of B atom at the specific point [atomic fraction]
    T : TYPE float
        DESCRIPTION. Temperature of the material [K]
    La : TYPE float 
        DESCRIPTION.atomic interaction constante [J/mol]

    Returns
    -------
    chem_pot : TYPE float
        DESCRIPTION. Chemical free energy density

    """
    chem_pot= R*T*(c*np.log(c)+(1-c)*np.log(1-c))+La*c*(1-c)
    return chem_pot




def plot_chemical_free_energy_density(c0,
                                      La):
    """
    Function to plot the chemical free energy density in function of composition 
    for c between [0,1] and T=673K to show where the initial composition of B atoms c0 stands 

    Parameters
    ----------
    c0 : TYPE float
        DESCRIPTION. average composition of B atom [atomic fraction]
    La : TYPE float 
        DESCRIPTION.Atom intaraction constant [J/mol]

    Returns
    -------
    fig : TYPE matplotlib.figure.Figure()
        DESCRIPTION. figure of the Chemical free energy density in function of composition
        for c between [0,1] and T=673K

    """
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot()
    
    T=673 
    cc = np.linspace(0.01, 0.99, 100)
    
    ax.plot(cc,chemical_free_energy_density(cc,T,La),color='black')
    ax.plot(c0, chemical_free_energy_density(c0,T,La),color='r',marker='o',markersize=10)
    
    ax.set_xlabel('composition c [at. frac]')
    ax.set_ylabel('Chemical free energy density')
    ax.set_title('Chemical free energy density in function of composition \n for c between [0,1] and T=673K')
    
    return fig


def boundary_conditions(x,
                        y,
                        Nx,
                        Ny):
    """
    Function to define the boundary conditions of the Nx*Ny matrix. If a border is reached, one goes 
    to the border on the oposite side. 

    Parameters
    ----------
    x : TYPE int
        DESCRIPTION. coordinate of the grid cell along the x direction 
    y : TYPE int
        DESCRIPTION. coordinate of the grid cell along the y direction 
    Nx : TYPE int
        DESCRIPTION. number of computational grids along the x direction
    Ny : TYPE int 
        DESCRIPTION. number of computational grids along the y direction

    Returns
    -------
    x : TYPE int 
        DESCRIPTION. coordinate of the grid cell along the x direction 
    y : TYPE int 
        DESCRIPTION. coordinate of the grid cell along the y direction 
    x_plus : TYPE int
        DESCRIPTION. coordinate of the grid cell on the right along the x direction
    x_min : TYPE int
        DESCRIPTION. coordinate of the grid cell on the left along the x direction
    y_plus : TYPE int
        DESCRIPTION. coordinate of the grid cell above along the y direction
    y_min : TYPE int 
        DESCRIPTION.coordinate of the grid cell under along the y direction

    """
    #Renaming the coordinates
    x=x
    y=y
    x_plus=x+1
    x_min=x-1
    y_plus=y+1
    y_min=y-1
    
    #periodic boundary conditions 

    if y_plus > Ny-1:
        y_plus = y_plus - Ny
    if x_min < 0:
        x_min= x_min + Nx
    if x_plus > Nx-1:
        x_plus = x_plus - Nx
    if y_min < 0:
        y_min =y_min + Ny
        
    return x,y,x_plus,x_min,y_plus,y_min


def composition_nearest_neighbours(c,
                                     x,
                                     y,
                                     Nx,
                                     Ny):
    """
    Function that returns the composition of B atoms of the cells around the central cell 
    at coordinates (x,y)
    
    Parameters
    ----------
    c : TYPE array
        DESCRIPTION. Composition matrix where c[x,y] is the composition at point of coordinate (x,y)
    x : TYPE int 
        DESCRIPTION. coordinate of the grid cell along the x direction 
    y : TYPE int
        DESCRIPTION. coordinate of the grid cell along the y direction 
    Nx : TYPE int
        DESCRIPTION. number of computational grids along the x direction
    Ny : TYPE int 
        DESCRIPTION. number of computational grids along the y direction

    Returns
    -------
    c_center : TYPE float
        DESCRIPTION. composition at point of coordinate (x,y) [atomic fraction]
    c_left : TYPE float
        DESCRIPTION. composition at point of coordinate (x-1,y) [atomic fraction]
    c_right : TYPE float
        DESCRIPTION. composition at point of coordinate (x+1,y) [atomic fraction]
    c_up : TYPE float
        DESCRIPTION. composition at point of coordinate (x,y+1) [atomic fraction]
    c_down : TYPE float
        DESCRIPTION. composition at point of coordinate (x,y-1) [atomic fraction]

    """
    #setting the coordinates of the cells to follow the boundary conditions
    x,y,x_plus,x_min,y_plus,y_min = boundary_conditions(x,y,Nx,Ny)

    #positions of the order parameters values around the center 
    c_center= c[x,y]
    c_left= c[x_min,y]
    c_right= c[x_plus,y]
    c_up= c[x,y_plus]
    c_down= c[x,y_min]
    
    return c_center,c_left,c_right,c_up,c_down



def diffusion_potential_chemical(cc,
                                 T,
                                 La):
    """
    function to calculate the chemical component of the diffusion potential.

    Parameters
    ----------
    cc : TYPE float
        DESCRIPTION. composition of the center cell [atomic fraction]
    T : TYPE float
        DESCRIPTION. temperature of the material [K]
    La : TYPE float 
        DESCRIPTION.Atom intaraction constant [J/mol]

    Returns
    -------
    mu_chem_dir : TYPE float 
        DESCRIPTION. chemical component of the diffusion potential

    """
    #chemical term of the diffusion potential
    mu_chem_dir= R*T*(np.log(cc)-np.log(1.0-cc))+ La*(1.0-2.0*cc)
    return mu_chem_dir


# def test_diffusion_potential_chemical():
#     """
#     test function for the function to calculate the chemical component of the diffusion potential

#     Returns
#     -------
#     None.

#     """
#     # Test case 1
#     cc = 0.5
#     T = 300
#     La = 10
    
#     expected_output = R*T*(np.log(cc)-np.log(1.0-cc))+ La*(1.0-2.0*cc)
#     assert np.isclose(diffusion_potential_chemical(cc, T, La), expected_output)

#     # Test case 2
#     cc = 0.8
#     T = 500
#     La = 5
#     expected_output = R*T*(np.log(cc)-np.log(1.0-cc))+ La*(1.0-2.0*cc)
#     assert np.isclose(diffusion_potential_chemical(cc, T, La), expected_output)

#     # Test case 3
#     cc = 0.3
#     T = 400
#     La = 8
#     expected_output = R*T*(np.log(cc)-np.log(1.0-cc))+ La*(1.0-2.0*cc)
#     assert np.isclose(diffusion_potential_chemical(cc, T, La), expected_output)
    

    
def total_diffusion_potential(c,
                              x,
                              y,
                              A,
                              dx,
                              dy,
                              T,
                              La,
                              Nx,
                              Ny):
    """
    Function to calculate the total diffusion potential potential, which is the sum of the chemical
    term and the gradient term for a given cell at (x,y). 

    Parameters
    ----------
    c : TYPE array
        DESCRIPTION. Composition matrix where c[x,y] is the composition at point of coordinate (x,y)
    x : TYPE int 
        DESCRIPTION. coordinate of the grid cell along the x direction 
    y : TYPE int
        DESCRIPTION. coordinate of the grid cell along the y direction 
    A : TYPE float
        DESCRIPTION. gradient coefficient [Jm2/mol]
    dx : TYPE float
        DESCRIPTION. spacing of the computational grid in the x direction [m]
    dy : TYPE float
        DESCRIPTION. spacing of the computational grid in the y direction [m]
    T : TYPE float
        DESCRIPTION. temperature of the material [K]
    La : TYPE float 
        DESCRIPTION.Atom intaraction constant [J/mol]
    Nx : TYPE int
        DESCRIPTION. number of computational grids along the x direction
    Ny : TYPE int 
        DESCRIPTION. number of computational grids along the y direction

    Returns
    -------
    mu_tot_dir : TYPE float 
        DESCRIPTION. total diffusion potential potential for the cell in (x,y)

    """
    
    #getting the composition of the neighrest neighbours around the point in (x,y)
    c_center,c_left,c_right,c_up,c_down = composition_nearest_neighbours(c,x,y,Nx,Ny)

    #laplacian of the chemical potential
    
    mu_grad_dir=-A*(func_laplacian(c_center,c_left,c_right,c_up,c_down,dx,dy))
    #chemical potential in one direction given by the position of the center
    
    mu_chem_dir=diffusion_potential_chemical(c_center,T,La)
    
    #total chemical potential in that direction 
    mu_tot_dir=mu_grad_dir + mu_chem_dir
    
    return mu_tot_dir
    
# def test_total_diffusion_potential():
#     # Test case 1
#     c = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])
#     x = 1
#     y = 1
#     A = 0.1
#     dx = 0.5
#     dy = 0.5
#     T = 300
#     La = 10
#     Nx = 3
#     Ny = 3
    
#     #using already tested functions
#     c_center,c_left,c_right,c_up,c_down = composition_nearest_neighbours(c,x,y,Nx,Ny)
#     expected_output = -A*(func_laplacian(c_center,c_left,c_right,c_up,c_down,dx,dy)) + diffusion_potential_chemical(c_center,T,La)

#     assert np.isclose(total_diffusion_potential(c, x, y, A, dx, dy, T, La, Nx, Ny), expected_output)

def update_order_parameter(c,
                           c_t,
                           Nx,
                           Ny,
                           A,
                           dx,
                           dy,
                           T,
                           La,
                           Diff_A,
                           Diff_B,
                           dt):
    """
    Function that updates the order parameter c (composition matrix) by solving the Cahn-Hilliard 
    equations at a time interval dt and naming it c_t. This will result in a two-dimensional phase-field simulation 
    of the spinodal decomposition.
    
    Parameters
    ----------
    c : TYPE array
        DESCRIPTION. Composition matrix where c[x,y] is the composition at point of coordinate (x,y)
    c_t : TYPE array
        DESCRIPTION. Composition matrix where c_t[x,y] is the composition at point of coordinate (x,y)
                        after a time dt
    Nx : TYPE int
        DESCRIPTION. number of computational grids along the x direction
    Ny : TYPE int 
        DESCRIPTION. number of computational grids along the y direction
    A : TYPE float
        DESCRIPTION. gradient coefficient [Jm2/mol]
    dx : TYPE float
        DESCRIPTION. spacing of the computational grid in the x direction [m]
    dy : TYPE float
        DESCRIPTION. spacing of the computational grid in the y direction [m]
    T : TYPE float
        DESCRIPTION. temperature of the material [K]
    La : TYPE float 
         DESCRIPTION.Atom intaraction constant [J/mol]
    Diff_A : TYPE float 
        DESCRIPTION.diffusion coefficient of atom A in [m2/s]
    Diff_B : TYPE float 
        DESCRIPTION.diffusion coefficient of atom B in [m2/s]
    dt : TYPE float 
        DESCRIPTION.time increment [s]

    Returns
    -------
    None
    
    """
    
    for j in range(Ny):
        for i in range(Nx):
            
            #Coordinates with boundary conditions
            x,y,x_plus,x_min,y_plus,y_min = boundary_conditions(i,j,Nx,Ny)
            
            #composition of the nearest neighbours
            c_center,c_left,c_right,c_up,c_down = composition_nearest_neighbours(c,x,y,Nx,Ny)
            
            #total diffusion potential for the differen directions 
            
            mu_center=total_diffusion_potential(c,x,y,A,dx,dy,T,La,Nx,Ny)
            
            mu_left=total_diffusion_potential(c,x_min,y,A,dx,dy,T,La,Nx,Ny)
            
            mu_right=total_diffusion_potential(c,x_plus,y,A,dx,dy,T,La,Nx,Ny)
            
            mu_up=total_diffusion_potential(c,x,y_plus,A,dx,dy,T,La,Nx,Ny)
            
            mu_down= total_diffusion_potential(c,x,y_min,A,dx,dy,T,La,Nx,Ny)
            
            #total chemical energy gradient
            nabla_mu=func_laplacian(mu_center,mu_left,mu_right,mu_up,mu_down,dx,dy)
    
            #Increments
            dc2dx2 = ((c_right-c_left)*(mu_right-mu_left))/(4*dx*dx)
            dc2dy2 = ((c_up-c_down)*(mu_up-mu_down))/(4*dy*dy)

            
            #Diffusion and mobility
            Diff_BA=Diff_B/Diff_A
            
            mobility = (Diff_A/R/T)*(c_center+Diff_BA*(1-c_center))*c_center*(1-c_center)
            
            dmdc = (Diff_A/R/T)*((1-Diff_BA)*c_center*(1-c_center)+(c_center+Diff_BA*(1-c_center))*(1-2*c_center))
            #writting the right hand side of the Cahn-Hilliard equation
            
            dcdt=mobility*nabla_mu + dmdc*(dc2dx2+dc2dy2)
    
            #updating the order parameter c following the equation
          
            c_t[x,y] = c[x,y] + dcdt * dt 
            
            

# def test_update_order_parameter():
#     """
#     test function t check if the order parameter is well updated 

#     Returns
#     -------
#     None.

#     """
#     # Test case 1

#     T = 673 # temperature [K]
#     La=13943
#     A= 3.0e-14 # gradient coefficient [Jm2/mol]
#     coef_DA=1.0e-04
#     coef_DB=2.0e-05
#     E_DA=300000.0
#     E_DB=300000.0 
#     Nx= 3 #number of computational grids along the x direction
#     Ny= 3 #number of computational grids along the y direction
#     dx =  2.0e-9 # spacing of computational grids [m]
#     dy =  2.0e-9 # spacing of computational grids [m]
#     Diff_A = diffusion_coeff(coef_DA,E_DA,T)# diffusion coefficient of A atom [m2/s]
#     Diff_B = diffusion_coeff(coef_DB,E_DB,T) # diffusion coefficient of B atom [m2/s]
#     dt = time_increment(dx,Diff_A)


#     c = np.array([[0.50548814, 0.50715189, 0.50602763],
#                   [0.50544883, 0.50423655, 0.50645894],
#                   [0.50437587, 0.50891773, 0.50963663]])
#     c_t = np.zeros_like(c)
    
#     expected_output = np.array([[0.50532619, 0.50676076, 0.50666109],
#                                 [0.50488973, 0.50528719, 0.50639366],
#                                 [0.50563731, 0.5081079,  0.50866898]])
#     update_order_parameter(c, c_t, Nx, Ny, A, dx, dy, T, La, Diff_A, Diff_B, dt)

#     assert np.allclose(c_t, expected_output)
    


def initial_composition(Nx,
                          Ny,
                          c0,
                          seed=None):
    """
    Function that sets the initial matrix for the composition c and c_t
    at zero. Then, adds a random initial composition of B atoms in a grid of size Nx*Ny 
    going from the initial composition c0 for the matrix c. 

    Parameters
    ----------
    Nx : TYPE int
        DESCRIPTION. number of computational grids along the x direction
    Ny : TYPE int 
        DESCRIPTION. number of computational grids along the y direction
    c0 : TYPE float
        DESCRIPTION. average composition of B atom [atomic fraction]
    seed : TYPE int or None, optional
        DESCRIPTION. random seed for reproducibility

    Returns
    -------
    c : TYPE list
        DESCRIPTION. Composition matrix where c[x,y] is the composition at point of coordinate (x,y)
                    with an initial random composition 
    c_t : TYPE list
        DESCRIPTION. Composition matrix where c_t[x,y] is the composition at point of coordinate (x,y)
                        after a time dt, set to zero

    """
    #Setting compositions matrix to 0
    c= np.zeros((Nx,Ny))
    c_t= np.zeros((Nx,Ny))
    
    #adding random fluctuations 
    c = add_fluctuation(Nx,Ny,c0,seed)
    
    return c,c_t



def plot_initial_composition(c):
    """
    Function to plot the initial composition.

    Parameters
    ----------
    c : TYPE list
        DESCRIPTION. Composition matrix where c[x,y] is the composition at point of coordinate (x,y)
                    with an initial random composition 

    Returns
    -------
    fig : TYPE matplotlib.figure.Figure()
        DESCRIPTION. Figure of the initial composition

    """
    #printing separatly the initial composition plot
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot()
    im=ax.imshow(c,cmap='bwr')
    ax.set_title('initial composition of B atoms')
    fig.colorbar(im,ax=ax)
    return fig

   