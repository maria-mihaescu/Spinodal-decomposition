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

from Binary_Alloys import interaction_parameter
from Binary_Alloys import set_free_energy
from Binary_Alloys import plot_anim_3d, plot_2d


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

# Define the window layout
layout = [
    [sg.Text("Free energy of a binary alloy in the Quasi-chemical atomistic model")],
    [sg.Text('Atomic number Z'), sg.InputText()],
    [sg.Text('Fraction of energy difference in eV'), sg.InputText()],
    [sg.Text('Temperature in K for calculation of G in (X_B, eta) space'), sg.InputText()],
    [sg.Canvas(key="-FIG0-"),sg.Canvas(key="-FIG1-")],
    [sg.Button("Ok"),sg.Button('Cancel')],
]


# Create the form and show it without the plot
window = sg.Window(
    "Free energy in function of composition for different T",
    layout,
    location=(0, 0),
    finalize=True,
    element_justification="center",
    font="Helvetica 18",
)

#Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
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
    draw_figure(window["-FIG0-"].TKCanvas, fig0)
    draw_figure(window["-FIG1-"].TKCanvas, fig1)

    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break
    
window.close()

