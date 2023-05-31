# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:46:02 2023

@author: maria
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation

#Goal of the simulation two-dimensional phase-field simulation 
#of the spinodal decomposition using Cahn-Hilliard equation.
#define the simulation cell parameters
Nx= 30 #number of computational grids along the x direction
Ny= 30 #number of computational grids along the y direction
NxNy = Nx*Ny
dx =  2.0e-9 # spacing of computational grids [m]
dy =  2.0e-9 # spacing of computational grids [m]

#parameters specific to the material

c0 = 0.5 # average composition of B atom [atomic fraction]

R = 8.314 # gas constant
T = 673 # temperature [K]

La = 20000.-9.*T # Atom intaraction constant [J/mol]
A= 3.0e-14 # gradient coefficient [Jm2/mol]

Diff_A = 1.0e-04*np.exp(-300000.0/R/T) # diffusion coefficient of A atom [m2/s]
Diff_B = 2.0e-05*np.exp(-300000.0/R/T) # diffusion coefficient of B atom [m2/s]
noise=0.01
dt = (dx*dx/Diff_A)*0.1 # time increment [s]


def add_fluctuation(Nx, Ny, c0, noise):  
    c = c0 + np.random.rand(Nx, Ny)*noise
    return c

def func_laplacian(center,left,right,up,down):
    lap=(right-2*center+left)/dx/dx + (up -2*center+down) /dy/dy
    return lap

def chemical_potential(c):
    chem_pot= R*T*(c*np.log(c)+(1-c)*np.log(1-c))+La*c*(1-c)
    return chem_pot

#Show the chemical potential in function of the order parameter

fig = plt.figure(figsize=(5,5))
cc = np.linspace(0.01, 0.99, 100);

plt.plot(cc, chemical_potential(cc),color='black')
plt.plot(c0, chemical_potential(c0),color='r',marker='o',markersize=10)
plt.xlabel('Concentration c [at. frac]')
plt.ylabel('Chemical free energy density')
plt.show()

def diffusion_potential_chemical(cc):
    #chemical term of the diffusion potential
    mu_chem_dir= R*T*(np.log(cc)-np.log(1.0-cc))+ La*(1.0-2.0*cc)
    return mu_chem_dir


def total_diffusion_potential(c,a,b):
    
    #Renaming the coordinates
    a_plus=a + 1
    a_min=a - 1
    b_plus=b + 1
    b_min=b - 1
    
    #periodic boundary conditions 

    if b_plus > Ny-1:
        b_plus = b_plus - Ny
    if a_min < 0:
        a_min= a_min + Nx
    if a_plus > Nx-1:
        a_plus = a_plus - Nx
    if b_min < 0:
        b_min =b_min + Ny
        
    #positions of the order parameters values around the center 
    c_center= c[a,b]
    c_left= c[a_min,b]
    c_right= c[a_plus,b]
    c_up= c[a,b_plus]
    c_down= c[a,b_min]
    
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
            
            #Renaming the coordinates
            x=i
            y=j
            x_plus=x+1
            x_min=x-1
            y_plus=y+1
            y_min=y-1
    
            #periodic boundary conditions 
            if y_plus > Ny-1:
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




#clear all starting point


"""
#printing separatly the initial concentration plot
plt.figure()
plt.imshow(c,cmap='bwr')
plt.title('initial concentration')
plt.colorbar()
plt.show()
"""

#Solving the Cahn-hiliard equation and plotting the result


# Plot NOT animated 
# Time integration parameters


c= np.zeros((Nx,Ny))
c_t= np.zeros((Nx,Ny))

c = c0 + np.random.rand(Nx, Ny)*0.01



# Plot animated 
# Time integration parameters


nsteps = 6000# total number of time-steps       
nprint = 600



#creating the snapshots list
fig, ax = plt.subplots()

snapshots=[]

for istep in range(1,nsteps+1):
    update_order_parameter(c,c_t)
    c[:,:]=c_t[:,:] # updating the order parameter every dt 
    
    if istep % nprint ==0:
        im = ax.imshow(c, cmap='bwr', animated=True)
        if istep == 1:
            ax.imshow(c, cmap='bwr')
    
        snapshots.append([im])       
    
anim = animation.ArtistAnimation(fig,snapshots,interval=50, blit=True,repeat_delay=1000)
                           





        

            