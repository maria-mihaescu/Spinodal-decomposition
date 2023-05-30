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

def func_laplacian(c,dx,dy):
    Nx,Ny =c.shape
    for i,j in zip(Nx,Ny):
        #nearest and second nearest neigbours 
        c_center = c[i,j]
        c_up= c[i,j+1]
        c_down=c[i,j-1]
        c_left=c[i-1,j]
        c_right=c[i+1,j]
        c_up_right=c[i+1,j+1]
        c_up_left=c[i-1,j+1]
        c_up_up=c[i,j+2]
        c_down_down=c[i,j-2]
        c_left_left=[i-2,j]
        c_right_right=c[i+2,j]
        c_down_left=c[i-1,j-1]
        c_down_right=c[i+1,j-1]
        
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

        
    
def add_fluctuation(Nx, Ny, c0, noise):
    c=c0+noise*(0.5-np.random.rand(Nx,Ny))
    return c
    
def chemical_potential(c,A):
   #use the analytic expression for the chemical potential obtained as
   #derivative of the homogeneous free energy
   mu=2*A*(c*((1-c)**2)-(c**2)*(1-c))
   return mu

def free_energy(c, A, k, dx, dy):
    Ny, Nx=c.shape
    c=c.tolist()
    Nx=int(Nx)
    Ny=int(Ny)
    c_top=c[Ny-1,:] + c[1:Ny,:]
    c_left=c[:,Nx] + c[:,1:Nx]
    
    C0=[]
    CL=[]
    CT=[]
    for c0,ct,cl in zip(c,c_top,c_left):
        c0_sum= [(c**2)*((1-c)**2) for c in c0]
        cl_sum=[(c1-c2)**2 for c1,c2 in zip(c0,cl)]
        ct_sum=[(c1-c2)**2 for c1,c2 in zip(c0,ct)]
        C0.append(np.sum(c0_sum))
        CL.append(np.sum(cl_sum))
        CT.append(np.sum(ct_sum))
    hom_term=dx*dy*A*np.sum(C0)
    grad_term = (k/dx)*np.sum(CL) + (k/dy)*np.sum(CT)
    F=hom_term+grad_term
    return F
        
# finite difference phase-field code to solve the cahn-hilliard equation

#clear all starting point
tstart = time.time()


#define the simulation cell parameters
Nx= 128
Ny= 128
NxNy = Nx*Ny
dx= 1
dy = 1

# Time integration parameters
nstep = 50000
nprint = 500
dtime = 1.0e-2
ttime = 0

#parameters specific to the material
c0=0.50
dc=0.02
mobility = 1.0
grad_coef = 0.5
A = 1.0 #multiplicative constant in free energy

#starting microstructure with a concentration fluctuation
c=add_fluctuation(Nx,Ny,c0,dc)


# initialization of the arrays that track the time evolution;
n2=round(nstep/nprint)+1
c_t=np.zeros(n2) # average concentration
F_t=np.zeros(n2) #free energy
t_t=np.zeros(n2) #time

c_mean=[]
for i in c:
    c_mean.append(np.mean(i))
c_t[1]=np.mean(c_mean)

F_t[1]=free_energy(c,A,grad_coef,dx,dy)


#Evolution of the cahn hillard equations
iplot=0

for istep in range(1,nstep):
    ttime=ttime+dtime #current time
    lap_c = func_laplacian(c,dx,dy) #laplacian of concentration
    mu_c  = chemical_potential(c,A) #chemical potential function;
    dF_dc = mu_c - 2 * grad_coef * lap_c #delta_F/delta_c in the free energy functional
    lap_dF_dc = func_laplacian(dF_dc,dx,dy) #laplacian of the above
    
    #time evolution
    c = c + dtime * mobility*lap_dF_dc
    print(c)
