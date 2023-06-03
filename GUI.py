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

from Binary_Alloys import interaction_parameter
from Binary_Alloys import set_free_energy
from Binary_Alloys import plot_anim_3d, plot_2d


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

# Define the window layout
def make_window1():
    layout = [
        [sg.Text("Free energy of a binary alloy in the Quasi-chemical atomistic model")],
        #[sg.Text('Atomic number Z'), sg.InputText(key='-IN-', enable_events=True)],
        #[sg.Text('Fraction of energy difference in eV'), sg.InputText(key='-IN-', enable_events=True)],
        #[sg.Text('Temperature in K for calculation of G in (X_B, eta) space'), sg.InputText(key='-IN-', enable_events=True)],
        [sg.Text('Atomic number Z'), sg.InputText()],
        [sg.Text('Fraction of energy difference in eV'), sg.InputText()],
        [sg.Text('Temperature in K for calculation of G in (X_B, eta) space'), sg.InputText()],
       
        [sg.Canvas(key="-FIG0-"),sg.Canvas(key="-FIG1-")],
        [sg.Button("Ok"),sg.Button('Next >')],
    ]

    return sg.Window(
        "Free energy in function of composition for different T",
        layout,
        location=(0, 0),
        finalize=True,
        element_justification="center",
        font="Helvetica 18")

def make_window2():
    layout = [[sg.Text('3D plots of the free energy in the (eta,X_B) space')],
              [sg.Button('Show')],
              [sg.Canvas(key="-3D_(eta,X_B)-")],
               [sg.Button('< Prev'), sg.Button('Next >')]]

    return sg.Window('3D plots of the free energy in the (eta,X_B) space', layout, finalize=True)

def make_window3():
    layout = [[sg.Text('3D plots of the free energy in the (X_B,T) space')],
              [sg.Button('Show')],
              [sg.Canvas(key="-3D_(X_B,T)-")],
               [sg.Button('< Prev'), sg.Button('Next >')]]

    return sg.Window('3D plots of the free energy in the (X_B,T) space', layout, finalize=True)

def make_window4():
    layout = [[sg.Text('3D plots of the free energy in the (eta,T) space')],
              [sg.Button('Show')],
              [sg.Canvas(key="-3D_(eta,T)-")],
               [sg.Button('< Prev'), sg.Button('Exit')]]

    return sg.Window('3D plots of the free energy in the (eta,T) space', layout, finalize=True)


window1, window2, window3, window4 = make_window1(), None, None, None

while True:
    
    window, event, values = sg.read_all_windows()

    if window==window1:
        
        Z=int(values[0])
        diff_eV=float(values[1])
        T0=float(values[2])
    
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
    
        
        # Add the plot to the window
        draw_figure(window1["-FIG0-"].TKCanvas, fig0)
        draw_figure(window1["-FIG1-"].TKCanvas, fig1)
            
        #if event == '-IN-' :
        #  window['-FIG0-'].update(draw_figure(window["-FIG0-"].TKCanvas, fig0))
         #  window["-FIG1-"].update(draw_figure(window["-FIG1-"].TKCanvas, fig1))

        if event == sg.WIN_CLOSED : # if user closes window
            break
        
        if event == 'Next >':
            window1.hide()
            window2 = make_window2()
            
            
    if window == window2:
        #Free energy surface in the (X_B,eta) space for temperature T=T0
        fig_3d_0=plot_anim_3d(X_XB_eta,eta_XB_eta,G_XB_eta,
                     'X_B','eta','G [ J/mole ]'
                     ,'G vs X_B and eta')
        
        draw_figure(window2["-3D_(eta,X_B)-"].TKCanvas, fig_3d_0)
        
        if event == sg.WIN_CLOSED : # if user closes window
            break
        
        if event == 'Next >':
            window2.hide()
            window3 = make_window3()
            
        elif event in (sg.WIN_CLOSED, '< Prev'):
            window2.hide()
            window1.un_hide()
            
    if window == window3:
        
        fig_3d_1=plot_anim_3d(X_XB_T,T_XB_T,G_XB_T,
                     'X_B','T [K]','G [ J/mole ]'
                     ,'G vs X_B and T, eta=0')
        # Free energy surface in (X_B,T) space for order parameter eta=0 
        draw_figure(window3["-3D_(X_B,T)-"].TKCanvas, fig_3d_1)
        
        if event == sg.WIN_CLOSED : # if user closes window
            break
        
        if event == 'Next >':
            window3.hide()
            window4 = make_window4()
            
        elif event in (sg.WIN_CLOSED, '< Prev'):
            window3.hide()
            window2.un_hide()


    if window == window4:
        #Free energy surface in (eta,T) space for equimolar composition (X_B=0.5)
        
        
        fig_3d_2=plot_anim_3d(eta_eta_T,T_eta_T,G_eta_T,
                     'eta','T [K]','G [ J/mole ]'
                     ,'G vs eta and T, X_B=0.5')
        
        draw_figure(window4["-3D_(eta,T)-"].TKCanvas, fig_3d_2)
        
        if event == sg.WIN_CLOSED or event == 'Exit': # if user closes window or clicks cancel
            break

        elif event in (sg.WIN_CLOSED, '< Prev'):
            window4.hide()
            window3.un_hide()
    

window.close()

