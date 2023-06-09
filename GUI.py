# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 18:51:10 2023

@author: Maria Mihaescu
"""

import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt 
import matplotlib.animation as animation

from Binary_Alloys import interaction_parameter
from Binary_Alloys import calculate_free_energy
from Binary_Alloys import plot_anim_3d, plot_2d

from Cahn_Hillard import diffusion_coeff
from Cahn_Hillard import time_increment
from Cahn_Hillard import plot_chemical_free_energy_density
from Cahn_Hillard import initial_composition
from Cahn_Hillard import plot_initial_composition
from Cahn_Hillard import update_order_parameter
from Cahn_Hillard import atom_interac_cst


def draw_figure(canvas,
                figure):
    """
    Function that draws the wanted figure in the empty canvas of the window

    Parameters
    ----------
    canvas : TYPE PySimpleGUI.Canvas(canvas, background_color, size)
        DESCRIPTION. drawable panel on the surface of the PySimpleGUI application window
    figure : TYPE matplotlib.figure.Figure()
        DESCRIPTION. figure we want to draw on the panel 

    Returns
    -------
    figure_canvas_agg : TYPE FigureCanvasTkAgg(figure, canvas)
        DESCRIPTION. figure drawn on the canvas

    """
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


def delete_fig_agg(fig_agg):
    """
    Function to delet the figure drawn on the canvas 

    Parameters
    ----------
    fig_agg : TYPE FigureCanvasTkAgg(figure, canvas)
        DESCRIPTION. figure drawn on the canvas

    Returns
    -------
    None.

    """
    fig_agg.get_tk_widget().forget()
    plt.close('all')
    
# Define the window layout

def make_window1():
    """
    Make the first window of the GUI

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    layout = [
        [sg.Text("Free energy of a binary alloy in the Quasi-chemical atomistic model")],
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
    """
    Make the second window of the GUI

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    layout = [[sg.Text('3D plots of the free energy in the different spaces')],
              [sg.Button('Show 3D plots')],
              [sg.Canvas(key="-3D_(eta,X_B)-"),sg.Canvas(key="-3D_(X_B,T)-")],
              [sg.Canvas(key="-3D_(eta,T)-")],
               [sg.Button('< Prev p1'), sg.Button('Next p3 >')]]

    return sg.Window('3D plots of the free energy in the different spaces', layout,location=(0, 0),
    finalize=True,
    element_justification="center")


def make_window3():
    """
    Make the third window of the GUI

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    layout = [[sg.Text('2D Spinodal decomposition solving Cahn Hilliards equations')],
              [sg.Text('Parameters specific to the material :')],
              [sg.Text('Average composition of atom B [at. frac]: c0'),sg.InputText(key='-IN_c0-')],
              [sg.Text('Temperature of the BiAlloy  [K] : T0'),sg.InputText(key='-IN_T0-')],
              [sg.Text('Gradient coefficient [J*M^2/mol] : A'),sg.InputText(key='-IN_A-')],
              
              [sg.Text('Coefficient for diffusion coefficient calculation of A atoms [M^2/s] : coef_DA'),sg.InputText(key='-IN_coef_DA-')],
              [sg.Text('Activation energy for diffusion coefficient of A atoms [J/mol] : E_DA'),sg.InputText(key='-IN_E_DA-')],
              [sg.Text('Coefficient for diffusion coefficient calculation of B atoms [M^2/s] : coef_DB'),sg.InputText(key='-IN_coef_DB-')],
              [sg.Text('Activation energy for diffusion coefficient of B atoms [J/mol] : E_DB'),sg.InputText(key='-IN_E_DB-')],
              
              [sg.Text('Parameters specific to the grid :')],
              [sg.Text('Number of computational grids along the x direction : Nx'),sg.InputText(key='-IN_Nx-')],
              [sg.Text('Number of computational grids along the y direction : Ny'),sg.InputText(key='-IN_Ny-')],
              [sg.Text('Spacing of computational grids along the x direction [M] : dx'),sg.InputText(key='-IN_dx-')],
              [sg.Text('Spacing of computational grids along the y direction [M] : dy'),sg.InputText(key='-IN_dy-')],
              
              [sg.Text('Parameters specific to the animation :')],
              [sg.Text('Total number of time-steps  : Nsteps'),sg.InputText(key='-IN_Nsteps-')],
              [sg.Text('Divisor of Nsteps used for printing : Nprint'),sg.InputText(key='-IN_Nprint-')],
              [sg.Text('Interval between each frame [ms]: Interval'),sg.InputText(key='-IN_Interval-')],
              
              [sg.Button('enter values'),sg.Button('< Prev p2'), sg.Button('Next p4 >')]]

    return sg.Window('Parameters for the spinodal decomposition solving Cahn Hilliards equations', layout, location=(0, 0),
    finalize=True, element_justification="center")

def make_window4():
    """
    Make the fourth window of the GUI

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    layout = [[sg.Text('Initial states:')],
              [sg.Button('Show initial plots')],
              [sg.Canvas(key="-c0_chemical_potential-"),sg.Canvas(key="-initial_composition-")],
               [sg.Button('< Prev p3'), sg.Button('Next p5 >')]]

    return sg.Window('Initial states of the spinodal decomposition solving Cahn Hilliards equations', layout, location=(0, 0),
    finalize=True, element_justification="center")

def make_window5():
    """
    Make the fifth window of the GUI

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    layout = [[sg.Text('Animation of the spinodal decomposition solving Cahn Hilliards equations in 2D:')],
              [sg.Button('Show animation')],
              [sg.Canvas(key="-anim-")],
               [sg.Button('< Prev p4'), sg.Button('Next p6 >')]]

    return sg.Window('Initial states of the spinodal decomposition solving Cahn Hilliards equations', layout, location=(0, 0),
    finalize=True, element_justification="center")

def make_window6():
    """
    Make the 6th window of the GUI

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    layout = [[sg.Text('Save the spinodal decomposition data')],
              [sg.Text('Path of the directory for the txt file of the composition data:'),sg.InputText(key='-IN_txt_path-')],
              [sg.Text('Name of the txt file:'),sg.InputText(key='-IN_txt_file-')],
              [sg.Button('Save txt')],
               [sg.Button('< Prev p5'), sg.Button('Exit')]]

    return sg.Window('Save the spinodal decomposition data', layout, location=(0, 0),
    finalize=True, element_justification="center")


#Make the first window and set the others windows to none 
window1, window2, window3, window4, window5, window6 = make_window1(), None, None, None, None, None

#Set the figure drawings to None in order to be able to update them each time 
figure_canvas_agg0 = None
figure_canvas_agg1 = None
figure_canvas_agg_3d0=None
figure_canvas_agg_3d1=None
figure_canvas_agg_3d2=None
figure_canvas_agg_chem_pot=None
figure_canvas_agg_init_c=None
figure_canvas_agg = None
#set the color bar to None in order to be able to update it later
cbar=None

while True:
    
    #read all the events, windows and values entered on the windows
    window,event,values = sg.read_all_windows()
    
    if window==window1:
              
        if event== sg.WIN_CLOSED : # if user closes window close the programm
            break
        
        elif event == 'Show Plots': #if user goes on the button show plots
            
            #set values to the entered user values
            Z=(int(values['-IN_Z-']))
            diff_eV=(float(values['-IN_X-']))
            T0=(float(values['-IN_T-']))

            
            #set parameters for the plots
            #ranges for composition, temperature and order parameter
            X_B=np.arange(0,1,0.01)      # Composition (chemical order parameter)
            T=np.arange(50,1000,50)      # Temperature space
            eta=np.arange(-0.5,0.5,0.01) # Order parameter (structural)
            
            omega = interaction_parameter(Z,diff_eV)
            
            #Set parameters for the graphs in all the different spaces
            X_XB_eta,eta_XB_eta,G_XB_eta,X_XB_T,T_XB_T,G_XB_T,eta_eta_T,T_eta_T,G_eta_T = calculate_free_energy(T0,omega)
             
            #set the 2D figures with those parameters
            fig0=plot_2d(X_B,G_XB_T,'X_B','G vs X_B for different T, eta=0')
            fig1=plot_2d(eta,G_eta_T,'eta','G vs eta for different T, X_B=0.5')

            #Delete the figures if they are already present 
            if figure_canvas_agg0 is not None:
                delete_fig_agg(figure_canvas_agg0)
            if figure_canvas_agg1 is not None:
                delete_fig_agg(figure_canvas_agg1)
                
            #Draw the computed figures on the empty canvas
            figure_canvas_agg0 = draw_figure(window["-FIG0-"].TKCanvas, fig0) 
            figure_canvas_agg1 = draw_figure(window["-FIG1-"].TKCanvas, fig1)


        elif event == 'Next p2 >':
            window1.hide()
            window2 = make_window2()
            
    if window == window2:
        
        if event == sg.WIN_CLOSED : # if user closes the window
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
            
            #Delete the figures if they are already present 
            if figure_canvas_agg_3d0 is not None:
                delete_fig_agg(figure_canvas_agg_3d0)
            if figure_canvas_agg_3d1 is not None:
                delete_fig_agg(figure_canvas_agg_3d1)
            if figure_canvas_agg_3d2 is not None:
                delete_fig_agg(figure_canvas_agg_3d2)
            
            #Draw the computed figures on the empty canvas
            figure_canvas_agg_3d0= draw_figure(window["-3D_(eta,X_B)-"].TKCanvas, fig_3d_0)
            figure_canvas_agg_3d1= draw_figure(window["-3D_(X_B,T)-"].TKCanvas, fig_3d_1)
            figure_canvas_agg_3d2= draw_figure(window["-3D_(eta,T)-"].TKCanvas, fig_3d_2)
            
           
        elif event == 'Next p3 >':
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
            c0=float(values['-IN_c0-'])
            T=float(values['-IN_T0-'])
            A=float(values['-IN_A-'])
            coef_DA=float(values['-IN_coef_DA-'])
            E_DA=float(values['-IN_E_DA-'])
            coef_DB=float(values['-IN_coef_DB-'])
            E_DB=float(values['-IN_E_DB-'])
            Nx=int(values['-IN_Nx-'])
            Ny=int(values['-IN_Ny-'])
            dx=float(values['-IN_dx-'])
            dy=float(values['-IN_dy-'])
            nsteps=int(values['-IN_Nsteps-'])
            nprint=int(values['-IN_Nprint-'])
            interval=float(values['-IN_Interval-'])
            
            #setting the complementary variables that are in function of the set ones
            La=atom_interac_cst(T) # Atom interaction constant [J/mol]
            Diff_A = diffusion_coeff(coef_DA,E_DA,T)# diffusion coefficient of A atom [m2/s]
            Diff_B = diffusion_coeff(coef_DB,E_DB,T) # diffusion coefficient of B atom [m2/s]
            dt = time_increment(dx,Diff_A)


        
        elif event == 'Next p4 >':
            window3.hide()
            window4 = make_window4()
           
            
        elif event == '< Prev p2':
            window3.close()
            window2.un_hide()


    if window == window4:
        if event== sg.WIN_CLOSED : # if user closes window 
            break
        
        elif event == 'Show initial plots': 
            
            """
            #Just for testing so we don't have to enter all those parameters each time
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
            dt = time_increment(dx,Diff_A)

            interval=400
            """
            
            #Defining a random initial composition
            c,c_t=initial_composition(Nx,Ny,c0)
            #set the figure with the initial data
            fig_chem_pot=plot_chemical_free_energy_density(c0,La)
            fig_init_c=plot_initial_composition(c)
            
            #Delete the figures if they are already present 
            if figure_canvas_agg_chem_pot is not None:
                delete_fig_agg(figure_canvas_agg_chem_pot)
            if figure_canvas_agg_init_c is not None:
                delete_fig_agg(figure_canvas_agg_init_c)

            #Draw the computed figures on the empty canvas
            figure_canvas_agg_chem_pot=draw_figure(window["-c0_chemical_potential-"].TKCanvas, fig_chem_pot)
            figure_canvas_agg_init_c=draw_figure(window["-initial_composition-"].TKCanvas, fig_init_c)
           
        
        elif event == 'Next p5 >':
            window4.hide()
            window5 = make_window5()

            
        elif event =='< Prev p3':        
            window4.close()
            window3.un_hide()

            
    if window == window5:
        if event == sg.WIN_CLOSED : # if user closes window or clicks cancel
            break
        
        elif event == 'Show animation': 
            fig = matplotlib.figure.Figure()
            ax = fig.add_subplot()
            
            #initialize the parameters and the lists
            snapshots=[]
            c_init=c
            current_time=0
            C_list=[c_init]
            Time=[current_time]

            
            for istep in range(1,nsteps+1):
                update_order_parameter(c,c_t,Nx,Ny,A,dx,dy,T,La,Diff_A,Diff_B,dt)
                c[:,:]=c_t[:,:] # updating the order parameter every dt 
                current_time=current_time+dt
                C_list.append(c)
                Time.append(current_time)
                
                if istep % nprint ==0:  
                    
                    #Delete the figures and colorbar if they are already present 
                    
                    if figure_canvas_agg is not None:
                        delete_fig_agg(figure_canvas_agg)
                        
                    if cbar is not None:
                        cbar.remove()
                        
                    im = ax.imshow(c, cmap='bwr', animated=True)
                    ax.set_title("composition of atom B at time {:.2f}".format(current_time))
                    cbar=fig.colorbar(im,ax=ax)
                    snapshots.append([im])   
                    
                    #Draw the computed figures on the empty canvas
                    figure_canvas_agg = draw_figure(window["-anim-"].TKCanvas, fig) 
                    window.Refresh()
                    
            anim = animation.ArtistAnimation(fig,snapshots,interval, blit=True,repeat_delay=10)

                          
        elif event == '< Prev p4':
            window5.close()
            window4.un_hide()
            
        elif event == 'Next p6 >':
            window5.hide()
            window6 = make_window6()

    if window == window6:
        
        if event == sg.WIN_CLOSED or event=='Exit': # if user closes window or presses exit
            break
        
        elif event =='Save txt':
            #set the values for the path and title
            path_txt=values['-IN_txt_path-']
            title_txt=values['-IN_txt_file-']
            
            path_to_file= path_txt + title_txt + '.txt'
            
            #create a new txt file on the desired path
            with open(path_to_file, 'w') as f:
                #For each composition matrix and each time
                for c,t in zip(C_list,Time):
                    f.write("composition at time {}".format(t))
                    f.write('\n')
                    #print the coordinates of the matrix cell and the composition of that cell
                    f.write('x'+','+'y'+','+'c[x,y]'+ "\n")
                    f.write('\n')
                    for x in range (Nx):
                        for y in range (Ny):
                            f.write(str(x)+','+str(y)+','+str(c[x,y])+ "\n")
                f.close()
                    
            
        elif event =='< Prev p5':
            window6.close()
            window5.un_hide()

        
window.close()
