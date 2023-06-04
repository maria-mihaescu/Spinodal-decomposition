# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:46:02 2023

@author: Maria Mihaescu
"""

import numpy as np
import matplotlib
import matplotlib.animation as animation

#Goal of the simulation two-dimensional phase-field simulation 
#of the spinodal decomposition using Cahn-Hilliard equation.
#define the simulation cell parameters

R = 8.314 # gas constant


#parameters specific to the material

c0 = 0.5 # average composition of B atom [atomic fraction]
T = 673 # temperature [K]
La = 20000.-9.*T # Atom interaction constant [J/mol]
A= 3.0e-14 # gradient coefficient [Jm2/mol]
Diff_A = 1.0e-04*np.exp(-300000.0/R/T) # diffusion coefficient of A atom [m2/s]
Diff_B = 2.0e-05*np.exp(-300000.0/R/T) # diffusion coefficient of B atom [m2/s]

#Parameters specific to the grid 
Nx= 32 #number of computational grids along the x direction
Ny= 32 #number of computational grids along the y direction
NxNy = Nx*Ny
dx =  2.0e-9 # spacing of computational grids [m]
dy =  2.0e-9 # spacing of computational grids [m]

#nsteps,nprint,interval

def time_increment(dx,Diff_A):
    # time increment [s]
    dt = (dx*dx/Diff_A)*0.1
    return dt

dt = time_increment(dx,Diff_A)

def add_fluctuation(Nx, Ny, c0, noise):  
    c = c0 + np.random.rand(Nx, Ny)*0.01
    return c

def func_laplacian(center,left,right,up,down):
    lap=(right-2*center+left)/dx/dx + (up -2*center+down) /dy/dy
    return lap

def chemical_potential(c):
    chem_pot= R*T*(c*np.log(c)+(1-c)*np.log(1-c))+La*c*(1-c)
    return chem_pot


def plot_chemical_potential(c0):
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot()
    cc = np.linspace(0.01, 0.99, 100);
    ax.plot(cc, chemical_potential(cc),color='black')
    ax.plot(c0, chemical_potential(c0),color='r',marker='o',markersize=10)
    ax.set_xlabel('Concentration c [at. frac]')
    ax.set_ylabel('Chemical free energy density')
    
    return fig

def boundary_conditions(x,y):
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

def concentration_nearest_neighbours(c,x,y):
    
    x,y,x_plus,x_min,y_plus,y_min = boundary_conditions(x, y)
        
    #positions of the order parameters values around the center 
    c_center= c[x,y]
    c_left= c[x_min,y]
    c_right= c[x_plus,y]
    c_up= c[x,y_plus]
    c_down= c[x,y_min]
    
    return c_center,c_left,c_right,c_up,c_down

def diffusion_potential_chemical(cc):
    #chemical term of the diffusion potential
    mu_chem_dir= R*T*(np.log(cc)-np.log(1.0-cc))+ La*(1.0-2.0*cc)
    return mu_chem_dir


def total_diffusion_potential(c,x,y):
    
    #getting the concentration of the neighrest neighbours around the point in (x,y)
    c_center,c_left,c_right,c_up,c_down = concentration_nearest_neighbours(c,x,y)

    #laplacian of the chemical potential
    
    mu_grad_dir=-A*(func_laplacian(c_center,c_left,c_right,c_up,c_down))
    #chemical potential in one direction given by the position of the center
    
    mu_chem_dir=diffusion_potential_chemical(c_center)
    
    #total chemical potential in that direction 
    mu_tot_dir=mu_grad_dir + mu_chem_dir
    
    return mu_tot_dir
    
def update_order_parameter(c,c_t):
    
    for j in range(Ny):
        for i in range(Nx):
            
            #Coordinates with boundary conditions
            x,y,x_plus,x_min,y_plus,y_min = boundary_conditions(i, j)
            
            #concentration of the nearest neighbours
            c_center,c_left,c_right,c_up,c_down = concentration_nearest_neighbours(c,x,y)
            
            #total diffusion potential for the differen directions 
            
            mu_center=total_diffusion_potential(c,x,y)
            
            mu_left=total_diffusion_potential(c,x_min,y)
            
            mu_right=total_diffusion_potential(c,x_plus,y)
            
            mu_up=total_diffusion_potential(c,x,y_plus)
            
            mu_down= total_diffusion_potential(c,x,y_min)
            
            #total chemical energy gradient
            nabla_mu=func_laplacian(mu_center,mu_left,mu_right,mu_up,mu_down)
    
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
def initial_concentration():
    #Setting concentrations to 0
    c= np.zeros((Nx,Ny))
    c_t= np.zeros((Nx,Ny))
    
    #adding random fluctuations 
    c = add_fluctuation(Nx, Ny, c0)
    
    return c,c_t

def plot_initial_concentration(c,c_t):
    #printing separatly the initial concentration plot
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot()
    im=ax.imshow(c,cmap='bwr')
    ax.set_title('initial concentration')
    fig.colorbar(im,ax=ax)
    return fig

def Cahn_hiliard_animated(c,c_t,nsteps,nprint,interval):
    
    # Plot animated 
    # Time integration parameters
    
    #creating the snapshots list with all the images
    
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot()
    cbar=None
    snapshots=[]
    
    for istep in range(1,nsteps+1):
        update_order_parameter(c,c_t)
        c[:,:]=c_t[:,:] # updating the order parameter every dt 
    
        if istep % nprint ==0:
            if cbar:
                cbar.remove()
            im = ax.imshow(c, cmap='bwr', animated=True)
            cbar=fig.colorbar(im,ax=ax)    
            snapshots.append([im])       
        
    anim = animation.ArtistAnimation(fig,snapshots,interval, blit=True,repeat_delay=10)
    return anim
    

"""
#run the programm 
c,c_t=initial_concentration()
plot_initial_concentration(c,c_t)
Cahn_hiliard_animated(c,c_t,600,60,700)                        
        
"""
            