from psychrometrics import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)




if False:
    ### ------------ Function for Use ------------ ###
    
    plot(T=None, w=None, phi=None, h=None, T_wet=None, v=None, P=101.325,
         T_lower=0.0, T_upper=50.0, w_lower=0.0, w_upper=0.03, n_dense=1001)
    



    ### ------------ Testing & Examples ------------ ###

    # Test: T=20, w=0.01, phi=0.685, h=45.48, T_wet=16.25, v=0.8438
    
    plot(T=20, w=0.01)
    plot(T=20, phi=0.685)
    plot(T=20, h=45.48)
    plot(T=20, T_wet=16.25)

    plot(w=0.01, phi=0.685)
    plot(phi=0.685, h=45.48)    


    
    states = {'T': [20, 25, 35],
              'w': [0.01, 0.015, 0.012]}
    plot(**states)



    states = {'T': [20, 25, 35],
              'w': [0.01, 0.015, 0.012],
              'P': 75.0}
    plot(**states)







def plot(*, T=None, w=None, phi=None, h=None, T_wet=None, v=None, P=101.325,
         T_lower=0.0, T_upper=50.0, w_lower=0.0, w_upper=0.03, n_dense=1001):
    
    if T is None:
        T = get_T(w=w, phi=phi, h=h, T_wet=T_wet, v=v, P=P)
    if w is None:
        w = get_w(T=T, phi=phi, h=h, T_wet=T_wet, v=v, P=P)
    T_states = np.atleast_1d(T)
    w_states = np.atleast_1d(w)
    T_states, w_states = np.broadcast_arrays(T_states, w_states)


    # create plot
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title('Psychrometric Chart @ Barometric Pressure: {} kPa'.format(P), fontsize=20)
    ax.set_ylabel('Humidity ratio (w) [kg moisture per kg dry air]', fontsize=14)
    ax.set_xlabel('Dry bulb temperature (T_dry) [degC]', fontsize=14)
    ax.grid(visible=True, which='both')

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.yaxis.set_major_locator(MultipleLocator(0.002))
    ax.yaxis.set_minor_locator(MultipleLocator(0.0005))

    #ax.set_xlim( (T_lower-2, T_upper+2) )
    #ax.set_ylim( (w_lower-0.001, w_upper+0.001) )


    # plot boundaries
    T_left_bound = np.linspace(T_lower, T_lower, 2)
    w_left_bound = np.linspace(w_lower, get_w(T=T_lower, phi=1, P=P), 2)

    T_right_bound = np.linspace(T_upper, T_upper, 2)
    w_right_bound = np.linspace(w_lower, w_upper, 2)

    T_bottom_bound = np.linspace(T_lower, T_upper, 2)
    w_bottom_bound = np.linspace(w_lower, w_lower, 2)

    T_top_bound = np.linspace(get_T(w=w_upper, phi=1, P=P), T_upper, 2)
    w_top_bound = np.linspace(w_upper, w_upper, 2)

    T_sat_bound = np.linspace(T_lower, get_T(w=w_upper, phi=1, P=P), n_dense)
    w_sat_bound = get_w(T=T_sat_bound, phi=1, P=P)  

    ax.plot(T_left_bound, w_left_bound, 'k')
    ax.plot(T_right_bound, w_right_bound, 'k')
    ax.plot(T_bottom_bound, w_bottom_bound, 'k')
    ax.plot(T_top_bound, w_top_bound, 'k')
    ax.plot(T_sat_bound, w_sat_bound, 'k')


    # plot iso-phi lines
    for i, phi in enumerate(np.linspace(0.1, 0.9, 9)): # per 0.1

        T_right = min(get_T(w=w_upper, phi=phi, P=P), T_upper)
        
        T = np.linspace(T_lower, T_right, n_dense)
        w = get_w(T=T, phi=phi, P=P)

        label = str(round(phi*100)) + '%'
        label += ' relative humidity (phi)' if (i == 0) else ''
        xlabel = T[2*n_dense//3]
        ylabel = w[2*n_dense//3]
        ax.text(xlabel, ylabel, label, fontsize=10, color='darkgreen')
        ax.plot(xlabel, ylabel, marker='+', color='darkgreen', markersize=5, markeredgewidth=1)
        
        ax.plot(T, w, linewidth=0.5, color='darkgreen')


    # plot iso-h lines
    h_lower = get_h(T=T_lower, w=w_lower, P=P)
    h_lower = np.ceil(h_lower)
    h_upper = get_h(T=T_upper, w=w_upper, P=P)
    h_upper = np.floor(h_upper)
    h_arr = np.arange(h_lower, h_upper+1, 10)

    i_label = int(0.75 * len(h_arr)) # which line shows long label

    for i, h in enumerate(h_arr): # per 10 [kJ per kg dry air]
        
        T_left = max(get_T(h=h, phi=1, P=P), T_lower)
        T_left = max(get_T(h=h, w=w_upper, P=P), T_left)
        T_right = min(get_T(h=h, w=0, P=P), T_upper)
        
        T = np.linspace(T_left, T_right, n_dense)
        w = get_w(T=T, h=h, P=P)

        label = str(round(h))
        label += ' enthalpy (h) [kJ per kg-dry-air]' if (i == i_label) else ''
        xlabel = T[2*n_dense//3]
        ylabel = w[2*n_dense//3]
        ax.text(xlabel, ylabel, label, fontsize=10, color='darkred')
        ax.plot(xlabel, ylabel, marker='+', color='darkred', markersize=5, markeredgewidth=1)
        
        ax.plot(T, w, linewidth=0.5, color='darkred')


    # plot iso-T_wet lines
    T_wet_lower = get_T_wet(T=T_lower, w=w_lower, P=P)
    T_wet_lower = np.ceil(T_wet_lower / 5) * 5
    T_wet_upper = get_T_wet(T=T_upper, w=w_upper, P=P)
    T_wet_upper = np.floor(T_wet_upper)
    T_wet_arr = np.arange(T_wet_lower, T_wet_upper+0.1, 1)

    i_label = int(0.75 * len(T_wet_arr) / 5) * 5 # which line shows long label
    
    for i, T_wet in enumerate(T_wet_arr): # per 1 [degC]
        
        T_left = max(get_T(T_wet=T_wet, phi=1, P=P), T_lower)
        T_left = max(get_T(T_wet=T_wet, w=w_upper, P=P), T_left)
        T_right = min(get_T(T_wet=T_wet, w=0, P=P), T_upper)
        
        T = np.linspace(T_left, T_right, n_dense)
        w = get_w(T=T, T_wet=T_wet, P=P)

        if i in range(0, len(T_wet_arr), 5):
            label = str(round(T_wet))
            label += ' web bulb temperature (T_wet) [degC]' if (i == i_label) else ''
            xlabel = T[n_dense//3]
            ylabel = w[n_dense//3]
            ax.text(xlabel, ylabel, label, fontsize=10)
            ax.plot(xlabel, ylabel, marker='+', markersize=5, markeredgewidth=1, color='k')
            
        ax.plot(T, w, 'k-.', linewidth=0.5)


    # plot iso-v lines
    v_lower = get_v(T=T_lower, w=w_lower, P=P)
    v_lower = np.ceil(v_lower/0.01) * 0.01
    v_upper = get_v(T=T_upper, w=w_upper, P=P)
    v_upper = np.floor(v_upper/0.01) * 0.01
    v_arr = np.arange(v_lower, v_upper+0.001, 0.01)
    
    i_label = int(0.75 * len(v_arr) / 2) * 2 # which line shows long label
    
    for i, v in enumerate(v_arr): # per 0.01 [m^3 per kg-dry-air]
        
        T_left = max(get_T(v=v, phi=1, P=P), T_lower)
        T_left = max(get_T(v=v, w=w_upper, P=P), T_left)
        T_right = min(get_T(v=v, w=0, P=P), T_upper)
        
        T = np.linspace(T_left, T_right, n_dense)
        w = get_w(T=T, v=v, P=P)

        if i in range(0, len(v_arr), 2):
            label = str(round(v, 2))
            label += ' specific volume (v) [m^3 per kg-dry-air]' if (i == i_label) else ''
            xlabel = T[n_dense//4]
            ylabel = w[n_dense//4]
            ax.text(xlabel, ylabel, label, fontsize=10, color='darkblue')
            ax.plot(xlabel, ylabel, marker='+', color='darkblue', markersize=5, markeredgewidth=1)
            
        ax.plot(T, w, 'darkblue', linewidth=0.5)




    # plot states
    ax.plot(T_states, w_states, color='k', linewidth=1, marker='x', markersize=10, markeredgewidth=2)
    if len(T_states) > 1:
        for i, (xlabel, ylabel) in enumerate(zip(T_states, w_states)):
            ax.text(xlabel, ylabel+0.0005, str(i), fontsize=16, color='k', ha='center')


    fig.show()



