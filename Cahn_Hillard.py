# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:46:02 2023

@author: maria
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import cm


def func_laplacian(c,dx,dy):
    Nx,Ny =c.shape
    c_top=c[Ny,:],c[1:Ny-1,:]
    c_bot=c[2:Ny,:],c[1,:]
    c_lef=c[:,Nx],c[:,1]
    c_rig=c[:,2:Nx],c[:,1]
    lap=(c_top+c_bot+c_rig+c_lef-4*c)/(dx*dy)
    return lap
    
def add_fluctuation(Nx, Ny, c0, noise):
    c=c0+noise*(0.5-np.random.rand(Nx,Ny))
    return c
    
def chemical_potential(c,A):
   #use the analytic expression for the chemical potential obtained as
   #derivative of the homogeneous free energy
   mu=[]
   for i in c:
       m=2*A*(i*((1-i)^2)-(i^2)*(1-i))
       mu.append(m)
   return mu

def free_energy(c, A, k, dx, dy):
    Ny, Nx=c.shape
    c_top=np.concatenate(c[Ny,:],c[1:Ny-1,:])
    c_left=np.concatenate(c[:,Nx], c[:,1:Nx-1])
    
    F=[]
    for c0,ct,cl in zip(c,c_top,c_left):
        hom_term=dx*dy*A*np.sum( np.sum( (c0^2)*((1-c0)^2)) )
        grad_term = (k/dx)*np.sum(np.sum ((c0-cl)^2) ) + (k/dy)*np.sum(np.sum ((c0-ct)^2))
        f=hom_term+grad_term
        F.append(f)
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
print(c)

# initialization of the arrays that track the time evolution;
n2=round(nstep/nprint)+1
c_t=np.zeros(n2) # average concentration
F_t=np.zeros(n2) #free energy
t_t=np.zeros(n2) #time

c_t[1] = np.mean(np.mean(c))
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
