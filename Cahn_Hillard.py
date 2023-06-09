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
    Function to calculate the time increment in s between each update of the concentration

    Parameters
    ----------
    dx : TYPE float
        DESCRIPTION. spacing of the computational grid in the x direction [m]
    Diff_A : TYPE float 
        DESCRIPTION.diffusion coefficient in [m2/s]

    Returns
    -------
    dt : TYPE float 
        DESCRIPTION.time increment [s]

    """
    dt = (dx*dx/Diff_A)*0.1
    return dt

def add_fluctuation(Nx, 
                    Ny, 
                    c0): 
    """
    function that adds random fluctuation of the concentration in a computational grid of size
    Nx*Ny, starting from an initial concentration of B atoms of c0 in [atomic fraction]

    Parameters
    ----------
    Nx : TYPE int
        DESCRIPTION. number of computational grids along the x direction
    Ny : TYPE int
        DESCRIPTION. number of computational grids along the y direction
    c0 : TYPE float
        DESCRIPTION. average composition of B atom [atomic fraction]

    Returns
    -------
    c : TYPE list 
        DESCRIPTION. Composition matrix where c[x,y] is the composition at point of coordinate (x,y)

    """
    c = c0 + np.random.rand(Nx, Ny)*0.01
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
    (concentration or diffusion potential) given the variation of the quantity 
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
    Function to calculate the chemical potential  in a specific point given a concentration of B atom, a temperature and 
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
    Function to plot the chemical free energy density in function of concentration 
    for c between [0,1] and T=673K to show where the initial concentration of B atoms c0 stands 

    Parameters
    ----------
    c0 : TYPE float
        DESCRIPTION. average composition of B atom [atomic fraction]
    La : TYPE float 
        DESCRIPTION.Atom intaraction constant [J/mol]

    Returns
    -------
    fig : TYPE matplotlib.figure.Figure()
        DESCRIPTION. figure of the Chemical free energy density in function of concentration
        for c between [0,1] and T=673K

    """
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot()
    
    T=673 
    cc = np.linspace(0.01, 0.99, 100)
    
    ax.plot(cc,chemical_free_energy_density(cc,T,La),color='black')
    ax.plot(c0, chemical_free_energy_density(c0,T,La),color='r',marker='o',markersize=10)
    
    ax.set_xlabel('Concentration c [at. frac]')
    ax.set_ylabel('Chemical free energy density')
    ax.set_title('Chemical free energy density in function of concentration \n for c between [0,1] and T=673K')
    
    return fig


def boundary_conditions(x,
                        y,
                        Nx,
                        Ny):
    """
    Function to define the boundary conditions of the Nx*Ny matrix. If a border is reached, one goes 
    to the border on the other side. 

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    Nx : TYPE
        DESCRIPTION.
    Ny : TYPE
        DESCRIPTION.

    Returns
    -------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    x_plus : TYPE
        DESCRIPTION.
    x_min : TYPE
        DESCRIPTION.
    y_plus : TYPE
        DESCRIPTION.
    y_min : TYPE
        DESCRIPTION.

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

def concentration_nearest_neighbours(c,
                                     x,
                                     y,
                                     Nx,
                                     Ny):
    """
    

    Parameters
    ----------
    c : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    Nx : TYPE
        DESCRIPTION.
    Ny : TYPE
        DESCRIPTION.

    Returns
    -------
    c_center : TYPE
        DESCRIPTION.
    c_left : TYPE
        DESCRIPTION.
    c_right : TYPE
        DESCRIPTION.
    c_up : TYPE
        DESCRIPTION.
    c_down : TYPE
        DESCRIPTION.

    """
    
    x,y,x_plus,x_min,y_plus,y_min = boundary_conditions(x, y,Nx,Ny)
        
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
    

    Parameters
    ----------
    cc : TYPE
        DESCRIPTION.
    T : TYPE
        DESCRIPTION.
    La : TYPE
        DESCRIPTION.

    Returns
    -------
    mu_chem_dir : TYPE
        DESCRIPTION.

    """
    #chemical term of the diffusion potential
    mu_chem_dir= R*T*(np.log(cc)-np.log(1.0-cc))+ La*(1.0-2.0*cc)
    return mu_chem_dir


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
    

    Parameters
    ----------
    c : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    A : TYPE
        DESCRIPTION.
    dx : TYPE
        DESCRIPTION.
    dy : TYPE
        DESCRIPTION.
    T : TYPE
        DESCRIPTION.
    La : TYPE
        DESCRIPTION.
    Nx : TYPE
        DESCRIPTION.
    Ny : TYPE
        DESCRIPTION.

    Returns
    -------
    mu_tot_dir : TYPE
        DESCRIPTION.

    """
    
    #getting the concentration of the neighrest neighbours around the point in (x,y)
    c_center,c_left,c_right,c_up,c_down = concentration_nearest_neighbours(c,x,y,Nx,Ny)

    #laplacian of the chemical potential
    
    mu_grad_dir=-A*(func_laplacian(c_center,c_left,c_right,c_up,c_down,dx,dy))
    #chemical potential in one direction given by the position of the center
    
    mu_chem_dir=diffusion_potential_chemical(c_center,T,La)
    
    #total chemical potential in that direction 
    mu_tot_dir=mu_grad_dir + mu_chem_dir
    
    return mu_tot_dir
    
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
    
#Goal of the simulation two-dimensional phase-field simulation 
#of the spinodal decomposition using Cahn-Hilliard equation.
    Parameters
    ----------
    c : TYPE
        DESCRIPTION.
    c_t : TYPE
        DESCRIPTION.
    Nx : TYPE
        DESCRIPTION.
    Ny : TYPE
        DESCRIPTION.
    A : TYPE
        DESCRIPTION.
    dx : TYPE
        DESCRIPTION.
    dy : TYPE
        DESCRIPTION.
    T : TYPE
        DESCRIPTION.
    La : TYPE
        DESCRIPTION.
    Diff_A : TYPE
        DESCRIPTION.
    Diff_B : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    for j in range(Ny):
        for i in range(Nx):
            
            #Coordinates with boundary conditions
            x,y,x_plus,x_min,y_plus,y_min = boundary_conditions(i,j,Nx,Ny)
            
            #concentration of the nearest neighbours
            c_center,c_left,c_right,c_up,c_down = concentration_nearest_neighbours(c,x,y,Nx,Ny)
            
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



#Solving the Cahn-hiliard equation and plotting the result
def initial_concentration(Nx,
                          Ny,
                          c0):
    """
    

    Parameters
    ----------
    Nx : TYPE
        DESCRIPTION.
    Ny : TYPE
        DESCRIPTION.
    c0 : TYPE
        DESCRIPTION.

    Returns
    -------
    c : TYPE
        DESCRIPTION.
    c_t : TYPE
        DESCRIPTION.

    """
    #Setting concentrations to 0
    c= np.zeros((Nx,Ny))
    c_t= np.zeros((Nx,Ny))
    
    #adding random fluctuations 
    c = add_fluctuation(Nx,Ny,c0)
    
    return c,c_t

def plot_initial_concentration(c):
    """
    

    Parameters
    ----------
    c : TYPE
        DESCRIPTION.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """
    #printing separatly the initial concentration plot
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot()
    im=ax.imshow(c,cmap='bwr')
    ax.set_title('initial concentration')
    fig.colorbar(im,ax=ax)
    return fig

def plot_concentration(c):
    """
    

    Parameters
    ----------
    c : TYPE
        DESCRIPTION.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """
    #printing separatly the initial concentration plot
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot()
    im=ax.imshow(c,cmap='bwr')
    ax.set_title('Concentration of B atoms')
    fig.colorbar(im,ax=ax)
    return fig

   