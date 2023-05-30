# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:46:02 2023

@author: maria
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import cm

#Goal of the simulation two-dimensional phase-field simulation 
#of the spinodal decomposition using Cahn-Hilliard equation.

def add_fluctuation(Nx, Ny, c0, noise):    
    c=c0+noise*(0.5-np.random.rand(Nx,Ny))
    return c

def func_laplacian(center,left,right,up,down,dx,dy):
    lap=(right-2*center+left)/(dx**2) + (up -2*center+down) /(dy**2)
    return lap
    
def chemical_potential(c_center,R,T,La):
    #use the analytic expression for the chemical potential obtained as
    #derivative of the homogeneous free energy
    mu_chem_dir= R*T*(np.log(c_center)-np.log(1-c_center))+La*(1-2*c_center)
    return mu_chem_dir

def total_diffusion_potential(c,x,y,R,T,A,La,dx,dy):
    c_center=c[x,y]
    c_left=[x-1,y]
    c_right=[x+1,y]
    c_up=[x,y+1]
    c_down=[x,y-1]
    
    mu_grad_dir=-A*func_laplacian(c_center,c_left,c_right,c_up,c_down,dx,dy)
    mu_chem_dir=chemical_potential(c_center,R,T,La)
    mu_tot_dir=mu_grad_dir + mu_chem_dir
    return mu_tot_dir
    
def update_order_parameter(c,c_t,R,T,A,La,Diff_A,Diff_B,dx,dy,dt):
    for i,j in zip(range(Nx+3),range(Ny+3)):
        #nearest and second nearest neigbours 
        c_center = c[i,j]
        c_up= c[i,j+1]
        c_down=c[i,j-1]
        c_left=c[i-1,j]
        c_right=c[i+1,j]
        
        
        c_up_up=c[i,j+2]
        c_down_down=c[i,j-2]
        c_left_left=[i-2,j]
        c_right_right=c[i+2,j]

        
        #periodic boundary conditions
        if c_up > c[i, Ny-1]:
            c_up = c[i,j+1-Ny]
        if c_left < c[0,j]:
            c_left= c[i-1+Nx,j]
        if c_right > c[Nx-1,j]:
            c_right = c[i+1-Nx,j]
        if c_down < c[i,0]:
            c_down=c[i,j-1+Ny]
            
        if c_up_up > c[i, Ny-1]:
            c_up_up = c[i,j+2-Ny]
        if c_left_left < c[0,j]:
            c_left_left= c[i-2+Nx,j]
        if c_right_right > c[Nx-1,j]:
            c_right_right = c[i+2-Nx,j]
        if c_down_down < c[i,0]:
            c_down_down=c[i,j-2+Ny]
        
        
        #total diffusion potential for the differen directions 
        mu_center=total_diffusion_potential(c,i,j,R,T,A,La,dx,dy)
        mu_left=total_diffusion_potential(c,i-1,j,R,T,A,La,dx,dy)
        mu_right=total_diffusion_potential(c,i+1,j,R,T,A,La,dx,dy)
        mu_up=total_diffusion_potential(c,i,j+1,R,T,A,La,dx,dy)
        mu_down= total_diffusion_potential(c,i,j-1,R,T,A,La,dx,dy)
        
        #total chemical energy gradient
        nabla_mu=func_laplacian(mu_center,mu_left,mu_right,mu_up,mu_down,dx,dy)

        print('\n nabla_mu=',nabla_mu)
        #Increments
        dc2dx2 = ((c_right-c_left)*(mu_right-mu_left))/(4*dx**2)
        dc2dy2 = ((c_up-c_down)*(mu_up-mu_down))/(4*dy**2)
        
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
        print('dcdt=', dcdt)
        dcdt=np.mean(dcdt)
        #updating the order parameter c following the equation
        print('dcdt=', dcdt)
        
        c_t[i,j] = c[i,j] + dcdt * dt 


#define the simulation cell parameters
Nx= 128 #number of computational grids along the x direction
Ny= 128 #number of computational grids along the y direction
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
nsteps = 50000 # total number of time-steps
nprint = 500
dt = (dx*dx/Diff_A)*0.1 # time increment [s]
ttime=0 #current time

#clear all starting point
c= np.zeros((Nx,Ny))
c_t= np.zeros((Nx,Ny))
noise=0.01

#starting microstructure with a concentration fluctuation
c=add_fluctuation(Nx,Ny,c0,noise)


#Solving the Cahn-hiliard equation 
for nstep in range(1,nsteps+1):
    update_order_parameter(c,c_t,R,T,A,La,Diff_A,Diff_B,dx,dy,dt)
    c[:,:]=c_t[:,:] # updating the order parameter every dt 
    ttime=ttime + dt #updating the current time 
    if np.mod(nstep,nprint)==0:
        plt.imshow(c,cmap='bwr')
        plt.title('Concentration of B atoms, nstep={}'.format(nstep))
        plt.colorbar()
        plt.show()
        