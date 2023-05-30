# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:46:02 2023

@author: maria
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as ani

#Goal of the simulation two-dimensional phase-field simulation 
#of the spinodal decomposition using Cahn-Hilliard equation.

def animate(c):
    ax.clear()
    ax.imshow(c,cmap='bwr')
    
def add_fluctuation(Nx, Ny, c0, noise):    
    c=c0+noise*(0.5-np.random.rand(Nx,Ny))
    return c

def func_laplacian(center,left,right,up,down,dx,dy):
    lap=(right-2*center+left)/(dx**2) + (up -2*center+down) /(dy**2)
    return lap
    
def chemical_potential(c_center,R,T,La):
    #chemical term of the diffusion potential
    mu_chem_dir= R*T*(np.log(c_center)-np.log(1-c_center))+La*(1-2*c_center)
    return mu_chem_dir

def total_diffusion_potential(c,x,y,R,T,A,La,dx,dy):
    #Renaming the coordinates
    x=x
    y=y
    x_plus=x+1
    x_min=x-1
    y_plus=y+1
    y_min=y-1
    
    #periodic boundary conditions 
    if y_plus> Ny-1:
        y_plus = y_plus-Ny
    if x_min< 0:
        x_min= x_min+Nx
    if x_plus > Nx-1:
        x_plus = x_plus -Nx
    if y_min< 0:
        y_min =y_min+Ny
        
    #positions of the order parameters values around the center 
    c_center=c[x,y]
    c_left=c[x_min,y]
    c_right=c[x_plus,y]
    c_up=c[x,y_plus]
    c_down=c[x,y_min]
    
    #laplacian of the chemical potential
    mu_grad_dir=-A*func_laplacian(c_center,c_left,c_right,c_up,c_down,dx,dy)
    #chemical potential in one direction given by the position of the center
    mu_chem_dir=chemical_potential(c_center,R,T,La)
    #total chemical potential in that direction 
    mu_tot_dir=mu_grad_dir + mu_chem_dir
    
    return mu_tot_dir
    
def update_order_parameter(c,c_t,R,T,A,La,Diff_A,Diff_B,dx,dy,dt):
    for j in range(Ny):
        for i in range(Nx):
            
            #Renaming the coordinates
            x=i
            y=j
            x_plus=i+1
            x_min=i-1
            y_plus=j+1
            y_min=y-1
    
            #periodic boundary conditions 
            if y_plus> Ny-1:
                y_plus = y_plus-Ny
            if x_min < 0:
                x_min= x_min + Nx
            if x_plus > Nx-1:
                x_plus = x_plus-Nx
            if y_min < 0:
                y_min =y_min+Ny
            
            #nearest neigbours 
            c_center = c[x,y]
            c_up= c[x,y_plus]
            c_down=c[x,y_min]
            c_left= c[x_min,y]
            c_right= c[x_plus,y]
     
            #total diffusion potential for the differen directions 
            mu_center=total_diffusion_potential(c,x,y,R,T,A,La,dx,dy)
            mu_left=total_diffusion_potential(c,x_min,y,R,T,A,La,dx,dy)
            mu_right=total_diffusion_potential(c,x_plus,y,R,T,A,La,dx,dy)
            mu_up=total_diffusion_potential(c,x,y_plus,R,T,A,La,dx,dy)
            mu_down= total_diffusion_potential(c,x,y_min,R,T,A,La,dx,dy)
            
            #total chemical energy gradient
            nabla_mu=func_laplacian(mu_center,mu_left,mu_right,mu_up,mu_down,dx,dy)
    
            print('\n nabla_mu=',nabla_mu)
            #Increments
            dc2dx2 = ((c_right-c_left)*(mu_right-mu_left))/(4*dx*dx)
            dc2dy2 = ((c_up-c_down)*(mu_up-mu_down))/(4*dy*dy)
            
            print('\n dc2dx2=',dc2dx2)
            print('\n dc2dy2=',dc2dy2)
            
            #Diffusion and mobility
            Diff_BA=Diff_B/Diff_A
            
            mobility = (Diff_A/R/T)*(c_center+Diff_BA*(1-c_center))*c_center*(1-c_center)
            print('mobility =',mobility)
            
            dmdc = (Diff_A/R/T)*((1-Diff_BA)*c_center*(1-c_center)+(c_center)+Diff_BA*(1-c_center))*(1-2*c_center)
            print('dmdc =', dmdc)
            #writting the right hand side of the Cahn-Hilliard equation
            dcdt=mobility*nabla_mu+dmdc*(dc2dx2+dc2dy2)
    
            #updating the order parameter c following the equation
            print('dcdt=', dcdt)
            
            c_t[i,j] = c[i,j] + dcdt * dt 


#define the simulation cell parameters
Nx= 30 #number of computational grids along the x direction
Ny= 30 #number of computational grids along the y direction
NxNy = Nx*Ny
dx = pow(10,-10) # spacing of computational grids [m]
dy = pow(10,-10) # spacing of computational grids [m]


#parameters specific to the material

c0 = 0.5 # average composition of B atom [atomic fraction]
R = 8.314 # gas constant
T = 673 # temperature [K]
La= 20000e-9*T # Atomic interaction constant [J/mol]
A= 3.0e-14 # gradient coefficient [Jm2/mol]
Diff_A = 1.0e-04*np.exp(-300000.0/R/T) # diffusion coefficient of A atom [m2/s]
Diff_B = 2.0e-05*np.exp(-300000.0/R/T) # diffusion coefficient of B atom [m2/s]

# Time integration parameters
nsteps = 11000 # total number of time-steps
nprint = 50
dt = (dx*dx/Diff_A)*0.1 # time increment [s]
ttime=0 #current time

#clear all starting point

c= np.zeros((Nx,Ny))
c_t= np.zeros((Nx,Ny))
noise=0.01


#starting microstructure with a concentration fluctuation
c=add_fluctuation(Nx,Ny,c0,noise)
plt.figure()
plt.imshow(c,cmap='bwr')
plt.title('initial concentration')
plt.colorbar()
plt.show()

#setting the frames

wframe = None
cbar=None
tstart = time.time()

#Solving the Cahn-hiliard equation and plotting the result
fig = plt.figure()
ax = fig.add_subplot()

for nstep in range(1,nsteps+1):
    update_order_parameter(c,c_t,R,T,A,La,Diff_A,Diff_B,dx,dy,dt)
    c[:,:]=c_t[:,:] # updating the order parameter every dt 
    ttime=ttime + dt #updating the current time 
    img=ax.imshow(c,cmap='bwr')
    animator=ani.FuncAnimation(img, update_order_parameter)
    cbar = plt.colorbar(img, ax=ax)
    title= ax.set_title('Concentration of B atoms at time {:.2f}'.format(ttime))
    plt.show()
    """
    if np.mod(nstep,nprint)==0:
        # If a line collection is already there remove it before drawing it again.
        if wframe:
            wframe.remove()
        #If there is already a color bar remove it before drawing it again
        if cbar:
            cbar.remove()
        #generate the data concentration data following the Cahn_Hilliard equation
        
        update_order_parameter(c,c_t,R,T,A,La,Diff_A,Diff_B,dx,dy,dt)
        c[:,:]=c_t[:,:] # updating the order parameter every dt 
        ttime=ttime + dt #updating the current time 
        
       # Plot the new surface plot
        img=ax.imshow(c,cmap='bwr')
        cbar = plt.colorbar(img, ax=ax)
        title= ax.set_title('Concentration of B atoms at time {:.2f}'.format(ttime))
        plt.show()
        """
        

            