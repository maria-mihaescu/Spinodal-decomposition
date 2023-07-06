# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 18:51:10 2023

@author: Maria Mihaescu
"""

import numpy as np
import h5py

import PySimpleGUI as sg

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.animation as animation

from Binary_Alloys import ranges

from Cahn_Hillard import diffusion_coeff
from Cahn_Hillard import time_increment
from Cahn_Hillard import initial_composition
from Cahn_Hillard import update_order_parameter
from Cahn_Hillard import atom_interac_cst

from GUI_Setup import load_configuration
from GUI_Setup import draw_figure
from GUI_Setup import delete_fig_agg

from GUI_Setup import setup_binary_alloy_fig2D
from GUI_Setup import setup_binary_alloy_fig3D
from GUI_Setup import setup_initial_composition_plots

    
# Define the window layout
def make_window0():
    """
    Make the initial window of the GUI
    in which the user can enter the configuration file
    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    layout = [
    [sg.Text('Configuration File Path:'), sg.Input(key='-FILE-'), sg.FileBrowse()],
    [sg.Button('Load Configuration'),sg.Button('Next p1 >')]]

    return sg.Window(
        "Configuration reader",
        layout,
        location=(0, 0),
        finalize=True,
        element_justification="center")

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
        [sg.Text('Number of nearest neighbours Z'), sg.InputText(key='-IN_Z-',default_text=Z_str)],
        [sg.Text('Fraction of energy difference in eV'), sg.InputText(key='-IN_X-',default_text=diff_eV_str)],
        [sg.Text('Temperature in [K] for calculation of G in (X_B, eta) space'), sg.InputText(key='-IN_T-',default_text=T0_str)],
        [sg.Text('Composition range, chemical order parameter range X :')]
        [sg.Text('Minimum of the composition range, X_B_min :'), sg.InputText(key='-IN_X_B_min-',default_text=X_B_min_str)],
        [sg.Text('Maximum of the composition range, X_B_max :'), sg.InputText(key='-IN_X_B_max-',default_text=X_B_max_str)],
        [sg.Text('Step of the composition range, X_B_step :'), sg.InputText(key='-IN_X_B_step-',default_text=X_B_step_str)],
        [sg.Text('Temperature range T:')]
        [sg.Text('Minimum of the temperature range, T_min :'), sg.InputText(key='-IN_T_min-',default_text=T_min_str)],
        [sg.Text('Maximum of the temperature range, T_max :'), sg.InputText(key='-IN_T_max-',default_text=T_max_str)],
        [sg.Text('Step of the temperature range, X_B_step :'), sg.InputText(key='-IN_T_step-',default_text=T_step_str)],
        [sg.Text('Structural order parameter range eta :')]
        [sg.Text('Minimum of the order parameter range, eta_min :'), sg.InputText(key='-IN_eta_min-',default_text=eta_min_str)],
        [sg.Text('Maximum of the order parameter range, eta_max :'), sg.InputText(key='-IN_eta_max-',default_text=eta_max_str)],
        [sg.Text('Step of the order parameter range, eta_step :'), sg.InputText(key='-IN_eta_step-',default_text=eta_step_str)],
   
        [sg.Canvas(key='-FIG0-'),sg.Canvas(key='-FIG1-')],
        [sg.Button('< Prev p0'),sg.Button('Show Plots'),sg.Button('Next p2 >')],
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
              [sg.Button('< Prev p1'), sg.Button('Next p3 >')],
              [sg.Canvas(key="-3D_(eta,X_B)-"),sg.Canvas(key="-3D_(X_B,T)-")],
              [sg.Canvas(key="-3D_(eta,T)-")]]

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
              [sg.Text('Average composition of atom B [at. frac]: c0'),sg.InputText(key='-IN_c0-',default_text=c0_str)],
              [sg.Text('Temperature of the BiAlloy  [K] : T0'),sg.InputText(key='-IN_T0-',default_text=T_str)],
              [sg.Text('Gradient coefficient [J*M^2/mol] : A'),sg.InputText(key='-IN_A-',default_text=A_str)],
              
              [sg.Text('Coefficient for diffusion coefficient calculation of A atoms [M^2/s] : coef_DA'),sg.InputText(key='-IN_coef_DA-',default_text=coef_DA_str)],
              [sg.Text('Activation energy for diffusion coefficient of A atoms [J/mol] : E_DA'),sg.InputText(key='-IN_E_DA-',default_text=E_DA_str)],
              [sg.Text('Coefficient for diffusion coefficient calculation of B atoms [M^2/s] : coef_DB'),sg.InputText(key='-IN_coef_DB-',default_text=coef_DB_str)],
              [sg.Text('Activation energy for diffusion coefficient of B atoms [J/mol] : E_DB'),sg.InputText(key='-IN_E_DB-',default_text=E_DB_str)],
              
              [sg.Text('Parameters specific to the grid :')],
              [sg.Text('Seed for the random generation of concentration'),sg.InputText(key='-IN_seed-',default_text=seed_str)],
             
              [sg.Text('Number of computational grids along the x direction : Nx'),sg.InputText(key='-IN_Nx-',default_text=Nx_str)],
              [sg.Text('Number of computational grids along the y direction : Ny'),sg.InputText(key='-IN_Ny-',default_text=Ny_str)],
              [sg.Text('Spacing of computational grids along the x direction [M] : dx'),sg.InputText(key='-IN_dx-',default_text=dx_str)],
              [sg.Text('Spacing of computational grids along the y direction [M] : dy'),sg.InputText(key='-IN_dy-',default_text=dy_str)],
              
              [sg.Text('Parameters specific to the animation :')],
              [sg.Text('Total number of time-steps  : Nsteps'),sg.InputText(key='-IN_Nsteps-',default_text=nsteps_str)],
              [sg.Text('Divisor of Nsteps used for printing : Nprint'),sg.InputText(key='-IN_Nprint-',default_text=nprint_str)],
              [sg.Text('Interval between each frame [ms]: Interval'),sg.InputText(key='-IN_Interval-',default_text=interval_str)],
              
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
              [sg.Text('Path of the directory for the HDF5 file of the composition data:'),sg.InputText(key='-IN_HDF5_path-')],
              [sg.Text('Name of the HDF5 file:'),sg.InputText(key='-IN_HDF5_file-')],
              [sg.Button('Save in HDF5')],
               [sg.Button('< Prev p5'), sg.Button('Exit')]]

    return sg.Window('Save the spinodal decomposition data', layout, location=(0, 0),
    finalize=True, element_justification="center")


#Make the first window and set the others windows to none 
window0, window1, window2, window3, window4, window5, window6 = make_window0(), None, None, None, None, None, None

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
    
    if window==window0:
        if event== sg.WIN_CLOSED : # if user closes window close the programm
            break
        
        elif event == 'Load Configuration':
            
            file_path = values['-FILE-']
            Names_config,Values_config=load_configuration(file_path)
            
            #assigning the str values from the configuration file
            
            #Binary Alloy
            Z_str=Values_config[0]
            diff_eV_str=Values_config[1]
            T0_str=Values_config[2]
            X_B_min_str=Values_config[3]
            X_B_max_str=Values_config[4]
            X_B_step_str=Values_config[5]
            eta_min_str=Values_config[6]
            eta_max_str=Values_config[7]
            eta_step_str=Values_config[8]
            T_min_str=Values_config[9]
            T_max_str=Values_config[10]
            T_step_str=Values_config[11]
            
            #Spinodal decomposition
            seed_str=Values_config[12]
            c0_str=Values_config[13]
            T_str=Values_config[14]
            A_str=Values_config[15]
            coef_DA_str=Values_config[16]
            E_DA_str=Values_config[17]
            coef_DB_str=Values_config[18]
            E_DB_str=Values_config[19]
            Nx_str=Values_config[20]
            Ny_str=Values_config[21]
            dx_str=Values_config[22]
            dy_str=Values_config[23]
            nsteps_str=Values_config[24]
            nprint_str=Values_config[25]
            interval_str=Values_config[26]
            
        elif event == 'Next p1 >':
            
            window0.hide()
            window1 = make_window1()
            
    if window==window1:
              
        if event== sg.WIN_CLOSED : # if user closes window close the programm
            break
        
        elif event == 'Show Plots': #if user goes on the button show plots
            
            #set values to the entered user values
            Z=(int(values['-IN_Z-']))
            diff_eV=(float(values['-IN_X-']))
            T0=(float(values['-IN_T-']))
            X_B_min=(float(values['-IN_X_B_min-']))
            X_B_max=(float(values['-IN_X_B_max-']))
            X_B_step=(float(values['-IN_X_B_step-']))
            eta_min=(float(values['-IN_eta_min-']))
            eta_max=(float(values['-IN_eta_max-']))
            eta_step=(float(values['-IN_eta_step-']))
            T_min=(float(values['-IN_T_min-']))
            T_max=(float(values['-IN_T_max-']))
            T_step=(float(values['-IN_T_step-']))
            
            #set parameters for the plots
            #ranges for composition, temperature and order parameter
            X_B,T,eta=ranges(X_B_min,X_B_max,X_B_step,T_min,T_max,T_step,eta_min,eta_max,eta_step)

            fig0, fig1= setup_binary_alloy_fig2D(Z,diff_eV,T0,X_B,eta)
            
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
            
        
        elif event == '< Prev p0':
            window1.close()
            window0.un_hide()
            
    if window == window2:
        
        if event == sg.WIN_CLOSED : # if user closes the window
            break
        
        elif event == 'Show 3D plots': 
            
            fig_3d_0, fig_3d_1, fig_3d_2= setup_binary_alloy_fig3D(Z,diff_eV,T0,X_B,eta)
            
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
            seed=int(values['-IN_seed-'])
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
            
            fig_chem_pot,fig_init_c = setup_initial_composition_plots(Nx,Ny,c0,La,seed)
            
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
            
            #Defining a random initial composition
            c,c_t=initial_composition(Nx,Ny,c0,seed)
            
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
        
        elif event =='Save in HDF5':
            
            #set the values for the path and title
            path_h5=values['-IN_HDF5_path-']
            title_h5=values['-IN_HDF5_file-']
            
            path_to_file= path_h5 + title_h5 + '.h5'
            
            with h5py.File(path_to_file, 'w') as file:
                for i, matrix in enumerate(C_list):
                    file.create_dataset(f'matrix_{i}', data=matrix)
                    
            
        elif event =='< Prev p5':
            window6.close()
            window5.un_hide()

        
window.close()
