# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:46:02 2023

@author: maria
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Goal of the simulation two-dimensional phase-field simulation 
#of the spinodal decomposition using Cahn-Hilliard equation.
#define the simulation cell parameters
Nx= 32 #number of computational grids along the x direction
Ny= 32 #number of computational grids along the y direction
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



#Setting concentrations to 0
c= np.zeros((Nx,Ny))
c_t= np.zeros((Nx,Ny))

#adding random fluctuations 
c = add_fluctuation(Nx, Ny, c0, noise)

#printing separatly the initial concentration plot
plt.figure()
plt.imshow(c,cmap='bwr')
plt.title('initial concentration')
plt.colorbar()
plt.show()


# Plot animated 
# Time integration parameters
nsteps = 600# total number of time-steps       
nprint = 60

#creating the snapshots list with all the images

fig, ax = plt.subplots()
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
    
anim = animation.ArtistAnimation(fig,snapshots,interval=500, blit=True,repeat_delay=10)
                           

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)



        

            