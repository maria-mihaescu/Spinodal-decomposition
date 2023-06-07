# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 18:51:10 2023

@author: Maria Mihaescu
"""

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import time

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt 

from Binary_Alloys import interaction_parameter
from Binary_Alloys import set_free_energy
from Binary_Alloys import plot_anim_3d, plot_2d

from Cahn_Hillard import diffusion_coeff
from Cahn_Hillard import time_increment
from Cahn_Hillard import plot_chemical_potential
from Cahn_Hillard import initial_concentration
from Cahn_Hillard import plot_initial_concentration
from Cahn_Hillard import Cahn_hiliard_animated
from Cahn_Hillard import update_order_parameter
from Cahn_Hillard import plot_concentration

#from Cahn_Hillard import atom_interac_cst


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


def delete_fig_agg(fig_agg):
    fig_agg.get_tk_widget().forget()
    plt.close('all')
    
# Define the window layout

def make_window1():
    layout = [
        [sg.Text("Free energy of a binary alloy in the Quasi-chemical atomistic model")],
        #[sg.Text('Atomic number Z'), sg.InputText(key='-IN-', enable_events=True)],
        #[sg.Text('Fraction of energy difference in eV'), sg.InputText(key='-IN-', enable_events=True)],
        #[sg.Text('Temperature in K for calculation of G in (X_B, eta) space'), sg.InputText(key='-IN-', enable_events=True)],
        [sg.Text('Atomic number Z'), sg.InputText(key='-IN_Z-')],
        [sg.Text('Fraction of energy difference in eV'), sg.InputText(key='-IN_X-')],
        [sg.Text('Temperature in [K] for calculation of G in (X_B, eta) space'), sg.InputText(key='-IN_T-')],
       
        [sg.Canvas(key='-FIG0-'),sg.Canvas(key='-FIG1-')],
        [sg.Button('Show Plots'),sg.Button('Next p2 >')],
    ]

    return sg.Window(
        "Free energy in function of composition for different T",
        layout,
        location=(0, 0),
        finalize=True,
        element_justification="center")

def make_window2():
    
    layout = [[sg.Text('3D plots of the free energy in the different spaces')],
              [sg.Button('Show 3D plots')],
              [sg.Canvas(key="-3D_(eta,X_B)-"),sg.Canvas(key="-3D_(X_B,T)-")],
              [sg.Canvas(key="-3D_(eta,T)-")],
               [sg.Button('< Prev p1'), sg.Button('Next p3 >')]]

    return sg.Window('3D plots of the free energy in the different spaces', layout,location=(0, 0),
    finalize=True,
    element_justification="center")

def make_window3():
    
    layout = [[sg.Text('2D Spinodal decomposition solving Cahn Hilliards equations')],
              [sg.Text('Parameters specific to the material :')],
              [sg.Text('Average composition of atom B [at. frac]: c0'),sg.InputText()],
              [sg.Text('Temperature of the BiAlloy  [K] : T0'),sg.InputText()],
              #[sg.Text('Atom interaction constant [J/mol] : La'),sg.InputText()],
              [sg.Text('Gradient coefficient [J*M^2/mol] : A'),sg.InputText()],
              
              [sg.Text('Coefficient for diffusion coefficient calculation of A atoms [M^2/s] : coef_DA'),sg.InputText()],
              [sg.Text('Activation energy for diffusion coefficient of A atoms [J/mol] : E_DA'),sg.InputText()],
              [sg.Text('Coefficient for diffusion coefficient calculation of B atoms [M^2/s] : coef_DB'),sg.InputText()],
              [sg.Text('Activation energy for diffusion coefficient of B atoms [J/mol] : E_DB'),sg.InputText()],
              
              [sg.Text('Parameters specific to the grid :')],
              [sg.Text('Number of computational grids along the x direction : Nx'),sg.InputText()],
              [sg.Text('Number of computational grids along the y direction : Ny'),sg.InputText()],
              [sg.Text('Spacing of computational grids along the x direction [M] : dx'),sg.InputText()],
              [sg.Text('Spacing of computational grids along the y direction [M] : dy'),sg.InputText()],
              
              [sg.Text('Parameters specific to the animation :')],
              [sg.Text('Total number of time-steps  : Nsteps'),sg.InputText()],
              [sg.Text('Divisor of Nsteps used for printing : Nprint'),sg.InputText()],
              [sg.Text('Interval between each frame [ms]: Interval'),sg.InputText()],
              
              [sg.Button('enter values'),sg.Button('< Prev p2'), sg.Button('Next p4 >')]]

    return sg.Window('Parameters for the spinodal decomposition solving Cahn Hilliards equations', layout, location=(0, 0),
    finalize=True, element_justification="center")

def make_window4():
    
    layout = [[sg.Text('Initial states:')],
              [sg.Button('Show initial plots')],
              [sg.Canvas(key="-c0_chemical_potential-"),sg.Canvas(key="-initial_concentration-")],
               [sg.Button('< Prev p3'), sg.Button('Next p5 >')]]

    return sg.Window('Initial states of the spinodal decomposition solving Cahn Hilliards equations', layout, location=(0, 0),
    finalize=True, element_justification="center")

def make_window5():
    
    layout = [[sg.Text('Animation of the spinodal decomposition solving Cahn Hilliards equations in 2D:')],
              [sg.Button('Show animation')],
              [sg.Canvas(key="-anim-")],
               [sg.Button('< Prev'), sg.Button('Exit')]]

    return sg.Window('Initial states of the spinodal decomposition solving Cahn Hilliards equations', layout, location=(0, 0),
    finalize=True, element_justification="center")


#Make the first window and set the others to none 
window1, window2, window3, window4, window5 = make_window1(), None, None, None, None

figure_canvas_agg0 = None
figure_canvas_agg1 = None
figure_canvas_agg_3d0=None
figure_canvas_agg_3d1=None
figure_canvas_agg_3d2=None

while True:
    
    window,event,values = sg.read_all_windows()
    
    if window==window1:
              
        if event== sg.WIN_CLOSED : # if user closes window
            break
        
        elif event == 'Show Plots':
            
            Z=(int(values['-IN_Z-']))
            diff_eV=(float(values['-IN_X-']))
            T0=(float(values['-IN_T-']))

            
            #set parameters for the plots
            #ranges for composition, temperature and order parameter
            X_B=np.arange(0,1,0.01)      # Composition (chemical order parameter)
            T=np.arange(50,1000,50)      # Temperature space
            eta=np.arange(-0.5,0.5,0.01) # Order parameter (structural)
            
            omega = interaction_parameter(Z,diff_eV)
            
            #Set parameters for the graphs
            X_XB_eta,eta_XB_eta,G_XB_eta,X_XB_T,T_XB_T,G_XB_T,eta_eta_T,T_eta_T,G_eta_T = set_free_energy(T0,omega)
             
            fig0=plot_2d(X_B,G_XB_T,'X_B','G vs X_B for different T, eta=0')
            fig1=plot_2d(eta,G_eta_T,'eta','G vs eta for different T, X_B=0.5')

            
            if figure_canvas_agg0 is not None:
                delete_fig_agg(figure_canvas_agg0)
            if figure_canvas_agg1 is not None:
                delete_fig_agg(figure_canvas_agg1)
                
            figure_canvas_agg0 = draw_figure(window1["-FIG0-"].TKCanvas, fig0) 
            figure_canvas_agg1 = draw_figure(window1["-FIG1-"].TKCanvas, fig1)


        elif event == 'Next p2 >':
            window1.hide()
            window2 = make_window2()
            
    if window == window2:
        
        if event == sg.WIN_CLOSED : # if user closes window
            break
        
        elif event == 'Show 3D plots': 
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
            
            if figure_canvas_agg_3d0 is not None:
                delete_fig_agg(figure_canvas_agg_3d0)
            if figure_canvas_agg_3d1 is not None:
                delete_fig_agg(figure_canvas_agg_3d1)
            if figure_canvas_agg_3d2 is not None:
                delete_fig_agg(figure_canvas_agg_3d2)

            figure_canvas_agg_3d0= draw_figure(window2["-3D_(eta,X_B)-"].TKCanvas, fig_3d_0)
            figure_canvas_agg_3d1= draw_figure(window2["-3D_(X_B,T)-"].TKCanvas, fig_3d_1)
            figure_canvas_agg_3d2= draw_figure(window2["-3D_(eta,T)-"].TKCanvas, fig_3d_2)
            
           
        elif event == 'Next p3 >':
            print('Next pushed')
            window2.hide()
            window3 = make_window3()

        elif event =='< Prev p1':
            window2.close()
            window1.un_hide()
            
    if window == window3:
        
        if event == sg.WIN_CLOSED : # if user closes window
            break
        
        elif event == 'enter values':
            #setting the variables to the user defined values
            c0=float(values[0])
            T=float(values[1])
            #La=float(values[2])
            A=float(values[2])
            coef_DA=float(values[3])
            E_DA=float(values[4])
            coef_DB=float(values[5])
            E_DB=float(values[6])
            Nx=int(values[7])
            Ny=int(values[8])
            dx=float(values[9])
            dy=float(values[10])
            nsteps=int(values[11])
            nprint=int(values[12])
            interval=float(values[13])
            
            #setting the complementary variables that are in function of the set ones
            La=20000.-9.*T # Atom interaction constant [J/mol]
            Diff_A = diffusion_coeff(coef_DA,E_DA,T)# diffusion coefficient of A atom [m2/s]
            Diff_B = diffusion_coeff(coef_DB,E_DB,T) # diffusion coefficient of B atom [m2/s]
            dt = time_increment(dx,Diff_A)
            print("Values stored")
            print(nsteps,nprint,interval,Nx,Ny,A,dx,dy,T,La,Diff_A,Diff_B,dt)
        
        elif event == 'Next p4 >':
            window3.hide()
            window4 = make_window4()
            #window=window4
            #event4, values4 = window4.read()
            
            
        elif event == '< Prev p2':
            window3.close()
            window2.un_hide()
            #window=window2

    if window == window4:
        if event== sg.WIN_CLOSED : # if user closes window 
            break
        
        elif event == 'Show initial plots': 
            def atom_interac_cst(T):    
                La = 20000.-9.*T # Atom interaction constant [J/mol]
                return La


            #parameters specific to the material entered by the user
            nsteps=600
            nprint=60
            c0 = 0.5 # average composition of B atom [atomic fraction]
            T = 673 # temperature [K]

            La=13943
            A= 3.0e-14 # gradient coefficient [Jm2/mol]

            coef_DA=1.0e-04
            coef_DB=2.0e-05
            E_DA=300000.0
            E_DB=300000.0


            #Parameters specific to the grid 
            Nx= 32 #number of computational grids along the x direction
            Ny= 32 #number of computational grids along the y direction
            dx =  2.0e-9 # spacing of computational grids [m]
            dy =  2.0e-9 # spacing of computational grids [m]
            
            Diff_A = diffusion_coeff(coef_DA,E_DA,T)# diffusion coefficient of A atom [m2/s]
            Diff_B = diffusion_coeff(coef_DB,E_DB,T) # diffusion coefficient of B atom [m2/s]

                        
            def time_increment(dx,Diff_A):
                # time increment [s]
                dt = (dx*dx/Diff_A)*0.1
                return dt
            
            dt = time_increment(dx,Diff_A)


            fig_chem_pot=plot_chemical_potential(c0,La)
            draw_figure(window["-c0_chemical_potential-"].TKCanvas, fig_chem_pot)
            
            #Defining a random initial concentration
            c,c_t=initial_concentration(Nx,Ny,c0)
            fig_init_c=plot_initial_concentration(c)
            draw_figure(window["-initial_concentration-"].TKCanvas, fig_init_c)
           
        
        elif event == 'Next p5 >':
            window4.hide()
            window5 = make_window5()
            #window=window5
            #event5, values5 = window5.read()
            
            
        elif event =='< Prev p4':
            
            window4.close()
            window3.un_hide()
            #window=window3
            
    if window == window5:
        if event == sg.WIN_CLOSED or event == 'Exit': # if user closes window or clicks cancel
            break
        
        elif event == 'Show animation': 
            figure_canvas_agg = None
            cbar=None
            for istep in range(1,nsteps+1):
                fig = matplotlib.figure.Figure()
                ax = fig.add_subplot() 
                update_order_parameter(c,c_t,Nx,Ny,A,dx,dy,T,La,Diff_A,Diff_B,dt)
                c[:,:]=c_t[:,:] # updating the order parameter every dt 
        
                if istep % nprint ==0:                    
                    if figure_canvas_agg:
                       figure_canvas_agg.get_tk_widget().forget()
                       plt.close('all')
                       #fig.clear()
                       #ax.cla()
                       #ax.clear()
                       #figure_canvas_agg = None
                    if cbar:
                        cbar.remove
                        
                    im = ax.imshow(c, cmap='bwr', animated=True)
                    cbar=fig.colorbar(im,ax=ax)
            
                    figure_canvas_agg = FigureCanvasTkAgg(fig,window["-anim-"].TKCanvas)
                    figure_canvas_agg.draw()
                    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
                    figure_canvas_agg
                    time.sleep(0.004)
                    window.Refresh()
                    
                    # figure_canvas_agg.get_tk_widget().forget()
                    # plt.close('all')
                    # fig.clear()
                    # figure_agg = None
                        
                    #if cbar:
                     #   cbar.remove()
                        

                          
        elif event == '< Prev p4':
            window5.close()
            window4.un_hide()
            #window=window4
        
        
window.close()
