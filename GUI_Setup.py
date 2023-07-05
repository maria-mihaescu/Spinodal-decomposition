# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 18:20:54 2023

@author: Maria Mihaescu
"""
from Binary_Alloys import interaction_parameter
from Binary_Alloys import free_energy_XB_eta
from Binary_Alloys import free_energy_XB_T
from Binary_Alloys import free_energy_eta_T
from Binary_Alloys import plot_anim_3d, plot_2d

from Cahn_Hillard import diffusion_coeff
from Cahn_Hillard import time_increment
from Cahn_Hillard import plot_chemical_free_energy_density
from Cahn_Hillard import initial_composition
from Cahn_Hillard import plot_initial_composition
from Cahn_Hillard import update_order_parameter
from Cahn_Hillard import atom_interac_cst

# Set-up of the first part on the Binary Allow

def setup_binary_alloy_fig2D(Z,
                             diff_eV,
                             T0,
                             X_B,
                             eta):
    """
    Function to setup the figures for the 2D representation of the binary alloy parameters in the GUI,
    with the user entered parameters. 

    Parameters
    ----------
    Z : TYPE int
        DESCRIPTION. number of nearest neighbours
    diff_eV : TYPE float
        DESCRIPTION. fraction of eV difference 
    T0 : TYPE float
        DESCRIPTION. temperature in K
    X_B : TYPE array
        DESCRIPTION. Composition (chemical order parameter)
    eta : TYPE array
        DESCRIPTION. Order parameter (structural)

    Returns
    -------
    fig0 : TYPE figure 
        DESCRIPTION. free energy in function of the composition for different temperatures
    fig1 : TYPE figure
        DESCRIPTION. free energy in function of the structural order parameter for different temperatures

    """
    #Setup the interaction parameter
    omega = interaction_parameter(Z,diff_eV)
    #Set parameters for the graphs in all the different spaces
    X_XB_eta,eta_XB_eta,G_XB_eta= free_energy_XB_eta(T0,omega)
    X_XB_T,T_XB_T,G_XB_T= free_energy_XB_T(T0,omega)
    eta_eta_T,T_eta_T,G_eta_T = free_energy_eta_T(T0,omega)
     
    #set the 2D figures with those parameters
    fig0=plot_2d(X_B,G_XB_T,'X_B','G vs X_B for different T, eta=0')
    fig1=plot_2d(eta,G_eta_T,'eta','G vs eta for different T, X_B=0.5')
    return fig0, fig1

def setup_binary_alloy_fig3D(Z,
                             diff_eV,
                             T0,
                             X_B,
                             eta):
    """
    Function to setup the figures for the 3D representation of the binary alloy parameters in the GUI,
    with the user entered parameters. 

    Parameters
    ----------
    Z : TYPE int
        DESCRIPTION. number of nearest neighbours
    diff_eV : TYPE float
        DESCRIPTION. fraction of eV difference 
    T0 : TYPE float
        DESCRIPTION. temperature in K
    X_B : TYPE array
        DESCRIPTION. Composition (chemical order parameter)
    eta : TYPE array
        DESCRIPTION. Order parameter (structural)

    Returns
    -------
    fig_3d_0 : TYPE figure 
        DESCRIPTION. figure of the free energy in function of the composition and the
        structural order parameter
    fig_3d_1 : TYPE figure 
        DESCRIPTION.figure of the free energy in function of the composition and the
        temperature
    fig_3d_2 : TYPE figure 
        DESCRIPTION. figure of the free energy in function of the temperature and the
        structural order parameter

    """
    #Setup the interaction parameter
    omega = interaction_parameter(Z,diff_eV)
    
    #Set parameters for the graphs in all the different spaces
    X_XB_eta,eta_XB_eta,G_XB_eta= free_energy_XB_eta(T0,omega)
    X_XB_T,T_XB_T,G_XB_T= free_energy_XB_T(T0,omega)
    eta_eta_T,T_eta_T,G_eta_T = free_energy_eta_T(T0,omega)
     
    #Free energy surface in the (X_B,eta) space for temperature T=T0
    fig_3d_0=plot_anim_3d(X_XB_eta,eta_XB_eta,G_XB_eta,
                 'X_B','eta','G [ J/mole ]'
                 ,'G vs X_B and eta')
    
    # Free energy surface in (X_B,T) space for order parameter eta=0 
    fig_3d_1=plot_anim_3d(X_XB_T,T_XB_T,G_XB_T,
                 'X_B','T [K]','G [ J/mole ]'
                 ,'G vs X_B and T, eta=0')
    
    #Free energy surface in (eta,T) space for equimolar composition (X_B=0.5)

    fig_3d_2=plot_anim_3d(eta_eta_T,T_eta_T,G_eta_T,
                 'eta','T [K]','G [ J/mole ]'
                 ,'G vs eta and T, X_B=0.5')
    return fig_3d_0,fig_3d_1,fig_3d_2

def setup_initial_composition_plots(Nx,
                              Ny,
                              c0,
                              La):
    """
    Function that return the plots and image describing the initial composition from which
    the spinodal decomposition starts from

    Parameters
    ----------
    Nx : TYPE int
        DESCRIPTION. number of computational grids along the x direction
    Ny : TYPE int
        DESCRIPTION. number of computational grids along the y direction
    c0 : TYPE float
        DESCRIPTION. average composition of B atom [atomic fraction]
    La : TYPE float 
        DESCRIPTION.atomic interaction constante [J/mol]

    Returns
    -------
    fig_chem_pot : figure
        DESCRIPTION. figure of the Chemical free energy density in function of composition
        for c between [0,1] and T=673K
    fig_init_c : TYPE figure
        DESCRIPTION. Figure of the initial composition

    """
    #Defining a random initial composition
    c,c_t=initial_composition(Nx,Ny,c0)
    #set the figure with the initial data
    fig_chem_pot=plot_chemical_free_energy_density(c0,La)
    fig_init_c=plot_initial_composition(c)
    
    return fig_chem_pot,fig_init_c