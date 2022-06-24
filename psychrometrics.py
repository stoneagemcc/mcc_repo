import numpy as np
from scipy.optimize import root

### Find any states given any 2 states

### State Variables
# T     : Dry-bulb Temperature [degC]
# w     : Humidity Ratio or Absolute Humidty [kg-water/kg-dry-air]
# phi   : Relative Humidity []
# h     : Enthalpy of Air-Vapor Mixture [kJ/kg-dry-air]
# T_wet : Wet-buld Temperature [degC]
# v     : Specific Volume [m3/kg-dry-air]

### Other Variables
# P     : atmospheric Pressure [kPa]
# P_v   : Partial Pressure of Vapor in Air-Vapor Mixture [kPa]
# P_g   : Satuation Pressure of Water at Temperature T [kPa]
# T_dew : Dew-point Temperature at  




if False:
    ### ------------ Functions for Use ------------ ###
    
    get_T(*, w=None, phi=None, h=None, T_wet=None, v=None, P=P_0)
    get_w(*, T=None, phi=None, h=None, T_wet=None, v=None, P=P_0)
    get_phi(*, T=None, w=None, h=None, T_wet=None, v=None, P=P_0)
    get_h(*, T=None, w=None, phi=None, T_wet=None, v=None, P=P_0)
    get_T_wet(*, T=None, w=None, phi=None, h=None, v=None, P=P_0)
    get_v(*, T=None, w=None, phi=None, h=None, T_wet=None, P=P_0)
    get_T_dew(w, P=P_0)




if False:
    ### ------------ Testing & Examples ------------ ###
    
    # Test: T=20, w=0.01, phi=0.685, h=45.48, T_wet=16.25, v=0.8438
    
    print('Test: T=20, w=0.01, phi=0.685, h=45.48, T_wet=16.25, v=0.8438')
    print()
    
    print('Test w:')
    print(get_w(T=20, phi=0.685))
    print(get_w(T=20, h=45.48))
    print(get_w(T=20, T_wet=16.25))
    print(get_w(T=20, v=0.8438))
    print(get_w(phi=0.685, h=45.48))
    print(get_w(phi=0.685, T_wet=16.25))
    print(get_w(phi=0.685, v=0.8438))
    print(get_w(h=45.48, T_wet=16.25)) # Problematic: h and T_wet are highly dependent
    print(get_w(h=45.48, v=0.8438))
    print(get_w(T_wet=16.25, v=0.8438))
    print()

    print('Test T:')
    print(get_T(w=0.01, phi=0.685))
    print(get_T(w=0.01, h=45.48))
    print(get_T(w=0.01, T_wet=16.25))
    print(get_T(w=0.01, v=0.8438))
    print(get_T(phi=0.685, h=45.48))
    print(get_T(phi=0.685, T_wet=16.25))
    print(get_T(phi=0.685, v=0.8438))
    print(get_T(h=45.48, T_wet=16.25)) # Problematic: h and T_wet are highly dependent
    print(get_T(h=45.48, v=0.8438))
    print(get_T(T_wet=16.25, v=0.8438))
    print()

    print('Test h:')
    print(get_h(T=20, w=0.01))
    print(get_h(T=20, phi=0.685))
    print(get_h(T=20, T_wet=16.25))
    print(get_h(T=20, v=0.8438))
    print(get_h(w=0.01, phi=0.685))
    print(get_h(w=0.01, T_wet=16.25))
    print(get_h(w=0.01, v=0.8438))
    print(get_h(phi=0.685, T_wet=16.25))
    print(get_h(phi=0.685, v=0.8438))
    print(get_h(T_wet=16.25, v=0.8438))
    print()

    # test array input
    phi_s = np.linspace(0.1, 1, 40).reshape(8,5)
    phi_s[1,1] = np.nan
    T_wet_s = 16.25
    T = get_T(phi=phi_s, T_wet=T_wet_s)
    w = get_w(phi=phi_s, T_wet=T_wet_s)
    h = get_h(phi=phi_s, T_wet=T_wet_s)

    # test using pandas and inputs.csv
    inp = pd.read_csv('inputs.csv', index_col=0, parse_dates=True)
    t_db = inp.t_db
    t_wb = inp.t_wb
    w = get_w(T=t_db, T_wet=t_wb) # -> pandas Series
    h = get_h(T=t_db, T_wet=t_wb)
    phi = get_phi(T=t_db, T_wet=t_wb)








### Constants
P_0    = 101.325   # Sea Level Standard Pressure [kPa]
R_u    = 8.314472  # Universal Gas Constant [kJ/kmol-K]
M_v    = 18.015268 # Molar Mass of Water [kg/kmol]
M_a    = 28.966    # Molar Mass of Dry Air [kg/kmol]
h_fg_0 = 2500.9    # Enthalpy of vaporization [kJ/kg] @ Temperature T = 0 [degC]
cp_a   = 1.005     # Specific Heat Capacity of Dry Air [kJ/kg-K] @ T = 300 [K]
cp_v   = 1.864     # Specific Heat Capacity of Water Vapor [kJ/kg-K] @ T = 300 [K]
cp_f   = 4.18      # Specific Heat Capacity of Liquid Water [kJ/kg-K] @ T = 300 [K]





### functions for use ###

def get_T(*, w=None, phi=None, h=None, T_wet=None, v=None, P=P_0):
    '''T: Dry-bulb Temperature [degC]'''
    if w is not None:
        if phi is not None:
            # T(w, phi)
            T = T__phi__w(phi, w, P)
        elif h is not None:
            # T(w, h)
            T = T__h__w(h, w)
        elif T_wet is not None:
            # T(w, T_wet)
            T = T__T_wet__w(T_wet, w, P)
        elif v is not None:
            # T(w, v)
            T = T__v__w(v, w, P)
    else:
        # find T by iteration
        T = no_T_w(phi=phi, h=h, T_wet=T_wet, v=v, P=P)
    return T


def get_w(*, T=None, phi=None, h=None, T_wet=None, v=None, P=P_0):
    '''w: Humidity Ratio or Absolute Humidty [kg-water/kg-dry-air]'''
    if T is None:
        # find T by iteration
        T = no_T_w(phi=phi, h=h, T_wet=T_wet, v=v, P=P)
    if phi is not None:
        # w(T, phi)
        w = w__phi__T(phi, T, P)
    elif h is not None:
        # w(T, h)
        w = w__h__T(h, T)
    elif T_wet is not None:
        # w(T, T_wet)
        w = w__T_wet__T(T_wet, T, P)
    elif v is not None:
        # w(T, v)
        w = w__v__T(v, T, P)
    return w


def get_phi(*, T=None, w=None, h=None, T_wet=None, v=None, P=P_0):
    '''phi: Relative Humidity []'''
    if w is not None:
        if h is not None:
            # phi(w, h)
            T = T__h__w(h, w)
        elif T_wet is not None:
            # phi(w, T_wet)
            T = T__T_wet__w(T_wet, w, P)
        elif v is not None:
            # phi(w, v)
            T = T__v__w(v, w, P)
    else:
        if T is None:
            # find T by iteration
            T = no_T_w(phi=None, h=h, T_wet=T_wet, v=v, P=P)
        if h is not None:
            # phi(T, h)
            w = w__h__T(h, T)
        elif T_wet is not None:
            # phi(T, T_wet)
            w = w__T_wet__T(T_wet, T, P)
        elif v is not None:
            # phi(T, v)
            w = w__v__T(v, T, P)
    return phi__T__w(T, w, P)


def get_h(*, T=None, w=None, phi=None, T_wet=None, v=None, P=P_0):
    '''h: Enthalpy of Air-Vapor Mixture [kJ/kg-dry-air]'''
    if w is not None:
        if phi is not None:
            # h(w, phi)
            T = T__phi__w(phi, w, P)
        elif T_wet is not None:
            # h(w, T_wet)
            T = T__T_wet__w(T_wet, w, P)
        elif v is not None:
            # h(w, v)
            T = T__v__w(v, w, P)
    else:
        if T is None:
            # find T by iteration
            T = no_T_w(phi=phi, h=None, T_wet=T_wet, v=v, P=P)
        if phi is not None:
            # h(T, phi)
            w = w__phi__T(phi, T, P)
        elif T_wet is not None:
            # h(T, T_wet)
            w = w__T_wet__T(T_wet, T, P)
        elif v is not None:
            # h(T, v)
            w = w__v__T(v, T, P)
    return h__T__w(T, w)


def get_T_wet(*, T=None, w=None, phi=None, h=None, v=None, P=P_0):
    '''T_wet: Wet-buld Temperature [degC]'''
    if w is not None:
        if phi is not None:
            # T_wet(w, phi)
            T = T__phi__w(phi, w, P)
        elif h is not None:
            # T_wet(w, h)
            T = T__h__w(h, w)
        elif v is not None:
            # T_wet(w, v)
            T = T__v__w(v, w, P)
    else:
        if T is None:
            # find T by iteration
            T = no_T_w(phi=phi, h=h, T_wet=None, v=v, P=P)
        if phi is not None:
            # T_wet(T, phi)
            w = w__phi__T(phi, T, P)
        elif h is not None:
            # T_wet(T, h)
            w = w__h__T(h, T)
        elif v is not None:
            # T_wet(T, v)
            w = w__v__T(v, T, P)
    return T_wet__T__w(T, w, P)


def get_v(*, T=None, w=None, phi=None, h=None, T_wet=None, P=P_0):
    '''v: Specific Volume [m3/kg-dry-air]'''
    if w is not None:
        if phi is not None:
            # v(w, phi)
            T = T__phi__w(phi, w, P)
        elif h is not None:
            # v(w, h)
            T = T__h__w(h, w)
        elif T_wet is not None:
            # v(w, T_wet)
            T = T__T_wet__w(T_wet, w, P)
    else:
        if T is None:
            # find T by iteration
            T = no_T_w(phi=phi, h=h, T_wet=T_wet, v=None, P=P)
        if phi is not None:
            # v(T, phi)
            w = w__phi__T(phi, T, P)
        elif h is not None:
            # v(T, h)
            w = w__h__T(h, T)
        elif T_wet is not None:
            # v(T, T_wet)
            w = w__T_wet__T(T_wet, T, P)
    return v__T__w(T, w, P)


def get_T_dew(w, P=P_0):
    '''
    T_dew: Dew-point Temperature [degC]
    by solving: P_sat(T_dew) = P_v(w)
    '''
    P_v = P_v__w(w, P)
    return T_sat(P_v)








### Fundamental Relations ###

# Relationships among: (w, P_v)
# w = m_v/m_a = (M_v/M_a) * P_v / P_a
def P_v__w(w, P):
    '''P_v(w)'''
    return P * w / (M_v/M_a + w)

def w__P_v(P_v, P):
    '''w(P_v)'''
    return (M_v/M_a) * (P_v / (P - P_v))

# Relationships among: (phi, T, w)
# phi = m_v/m_g = P_v/P_g
def phi__T__w(T, w, P):
    '''phi(T, w)'''
    return P_v__w(w, P) / P_sat(T)

def w__phi__T(phi, T, P):
    '''w(phi, T)'''
    P_v = phi * P_sat(T)
    return w__P_v(P_v, P)

def T__phi__w(phi, w, P):
    '''T(phi, w)'''
    P_g = P_v__w(w, P) / phi
    return T_sat(P_g)

# Relationships among: (h, T, w)
# h = cp_a * T + w * (h_fg_0 + cp_v * T)
def h__T__w(T, w):
    '''h(T, w)'''
    return cp_a*T + w*(h_fg_0 + cp_v*T)

def T__h__w(h, w):
    '''T(h, w)'''
    return (h - w*h_fg_0) / (cp_a + w*cp_v)

def w__h__T(h, T):
    '''w(h, T)'''
    return (h - cp_a*T) / (cp_v*T + h_fg_0)

# Relationships among: (T_wet, T, w)
# Approximated by Adiabatic Saturated Temperature
# Derived from Foundamental Equation of Enthalpy Balance
# From State(T, w) to State(T=T_wet, phi=1)
# Through Adiabatic Saturation Process
# h(T, w) + (w_wet - w) * (cp_f*T_wet) = h(T_wet, w_wet)
# where: w_wet = w(P_v=P_sat(T_wet))
def T__T_wet__w(T_wet, w, P):
    '''T(T_wet, w)'''
    P_g = P_sat(T_wet)
    w_wet = w__P_v(P_g, P)
    h_wet = h__T__w(T_wet, w_wet)
    h = h_wet - ( (w_wet - w) * (cp_f*T_wet) )
    return T__h__w(h, w)

def w__T_wet__T(T_wet, T, P):
    '''w(T_wet, T)'''
    P_g = P_sat(T_wet)
    w_wet = w__P_v(P_g, P)
    h_wet = h__T__w(T_wet, w_wet)
    return ( h_wet - (cp_a*T + w_wet*cp_f*T_wet) ) / (h_fg_0 + cp_v*T - cp_f*T_wet)

def T_wet__T__w(T, w, P):
    '''T_wet(T, w)'''
    T, w, P = np.broadcast_arrays(T, w, P)
    shape = T.shape
    T, w, P = [np.array(var, copy=True).ravel() for var in [T, w, P]]
    isna = np.isnan(T) | np.isnan(w) | np.isnan(P)
    for var in [T, w, P]:
        var[isna] = 1.0
    def func(T_wet):
        return T__T_wet__w(T_wet, w, P) - T
    T_wet = root(func, T).x # using scipy.optimize.root
    T_wet[isna] = np.nan
    return T_wet.reshape(shape) if bool(shape) else T_wet.item()

# Relationships among: (v, T, w)
# v = R_a * T_abs / P_a
def v__T__w(T, w, P):
    '''v(T, w)'''
    P_v = P_v__w(w, P)
    return (R_u/M_a) * (T + 273.15) / (P - P_v)

def T__v__w(v, w, P):
    '''T(v, w)'''
    P_v = P_v__w(w, P)
    return ( (P - P_v) * v / (R_u/M_a) ) - 273.15

def w__v__T(v, T, P):
    '''w(v, T)'''
    P_v = P - ( (R_u/M_a) * (T + 273.15) / v )
    return w__P_v(P_v, P)






### Using iteration for other relationships ###

def no_T_w(phi=None, h=None, T_wet=None, v=None, P=P_0):
    _phi, _h, _T_wet, _v, P = np.broadcast_arrays(phi, h, T_wet, v, P)
    shape = P.shape
    _phi, _h, _T_wet, _v, P = [np.array(var, copy=True).ravel()
                               for var in [_phi, _h, _T_wet, _v, P]]
    isna = np.isnan(P)
    for v0, v1 in zip([phi, h, T_wet, v], [_phi, _h, _T_wet, _v]):
        if v0 is not None:
            isna |= np.isnan(v1)
    for v0, v1 in zip([phi, h, T_wet, v], [_phi, _h, _T_wet, _v]):
        v1[isna] = 1.0
        
    if (phi is not None) and (h is not None):
        def func(T):
            return w__phi__T(_phi, T, P) - w__h__T(_h, T)
    elif (phi is not None) and (T_wet is not None):
        def func(T):
            return w__phi__T(_phi, T, P) - w__T_wet__T(_T_wet, T, P)
    elif (phi is not None) and (v is not None):
        def func(T):
            return w__phi__T(_phi, T, P) - w__v__T(_v, T, P)
    elif (h is not None) and (T_wet is not None):
        print('Problematic: h and T_wet are highly dependent')
        def func(T):
            return w__h__T(_h, T) - w__T_wet__T(_T_wet, T, P)
    elif (h is not None) and (v is not None):
        def func(T):
            return w__h__T(_h, T) - w__v__T(_v, T, P)
    elif (T_wet is not None) and (v is not None):
        def func(T):
            return w__T_wet__T(_T_wet, T, P) - w__v__T(_v, T, P)
        
    T = root(func, np.ones(P.shape)).x # using scipy.optimize.root
    T[isna] = np.nan
    return T.reshape(shape) if bool(shape) else T.item()







### Saturation Relations: P_sat(T) & T_sat(P) ###

def P_sat(T):
    '''
    input:
        T (degC) [-223.15 ≤ T ≤ 373.946(Critical Point)]
    output:
        P_sat (kPa)
    '''
    P_sat = np.where(T >= 0.01, P_sat_liquid(T), P_sat_ice(T))
    return P_sat if bool(np.shape(P_sat)) else P_sat.item()

def T_sat(P):
    '''
    input:
        P (kPa) [1.9349585E-43 ≤ P ≤ 22064.0(Critical Point)]
    output:
        T_sat (degC)
    '''
    T_sat = np.where(P >= 0.61121, T_sat_liquid(P), T_sat_ice(P))
    return T_sat if bool(np.shape(T_sat)) else T_sat.item()


def P_sat_liquid(T):
    '''
    Ref:
        IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
        Thermodynamic Properties of Water and Steam August 2007,
        http://www.iapws.org/relguide/IF97-Rev.html, Eq 30    
    input:
        T (degC) [0.01(Triple Point) ≤ T ≤ 373.946(Critical Point)]
    output:
        P_sat (kPa) 
    '''
    n = np.array([1.0,
                  0.11670521452767E+04, -0.72421316703206E+06, -0.17073846940092E+02,
                  0.12020824702470E+05, -0.32325550322333E+07, 0.14915108613530E+02,
                  -0.48232657361591E+04, 0.40511340542057E+06, -0.23855557567849E+00,
                  0.65017534844798E+03])
    T = T + 273.15
    tita = T + (n[9] / (T - n[10]))
    A = tita**2 + n[1]*tita + n[2]
    B = n[3]*tita**2 + n[4]*tita + n[5]
    C = n[6]*tita**2 + n[7]*tita + n[8]
    temp = 2*C / (-B + np.sqrt(B**2 - 4*A*C))
    P = 1000 * temp**4
    return P

def P_sat_ice(T):
    '''
    Ref:
        IAPWS, Revised Release on the Pressure along the Melting and Sublimation
        Curves of Ordinary Water Substance, http://iapws.org/relguide/MeltSub.html.
    input:
        T (degC) [-223.15 ≤ T ≤ 0.01(Triple Point)]
    output:
        P_sat (kPa)
    '''
    T_t, P_t = 273.16, 0.611657
    theta = (T + 273.15) / T_t
    a = np.array([-0.212144006e2, 0.273203819e2, -0.61059813e1])
    b = np.array([0.333333333e-2, 1.20666667, 1.70333333])
    log_P = (a[0]*theta**b[0] + a[1]*theta**b[1] + a[2]*theta**b[2]) / theta
    P = np.exp(log_P) * P_t
    return P

def T_sat_liquid(P):
    '''
    Ref:
        IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
        Thermodynamic Properties of Water and Steam August 2007,
        http://www.iapws.org/relguide/IF97-Rev.html, Eq 31   
    input:
        P (kPa) [0.61121(Triple Point) ≤ P ≤ 22064.0(Critical Point)]
    output:
        T_sat (degC)
    '''
    n = np.array([1.0,
                  0.11670521452767E+04, -0.72421316703206E+06, -0.17073846940092E+02,
                  0.12020824702470E+05, -0.32325550322333E+07, 0.14915108613530E+02,
                  -0.48232657361591E+04, 0.40511340542057E+06, -0.23855557567849E+00,
                  0.65017534844798E+03])
    beta = (P/1000)**0.25
    E = beta**2 + n[3]*beta + n[6]
    F = n[1]*beta**2 + n[4]*beta + n[7]
    G = n[2]*beta**2 + n[5]*beta + n[8]
    D = 2*G / (-F - np.sqrt(F**2 - 4*E*G) )
    temp = n[10] + D - np.sqrt( (n[10] + D)**2 - 4*(n[9] + n[10]*D) )
    T = temp / 2 - 273.15
    return T

def T_sat_ice(P):
    '''
    Reverse function of P_sat_ice(T)
    input:
        P (kPa) [1.9349585E-43 ≤ P ≤ 0.61121(Triple Point)]
    output:
        T_sat (degC)
    '''
    T_t, P_t = 273.16, 0.611657
    a = np.array([-0.212144006e2, 0.273203819e2, -0.61059813e1])
    b = np.array([0.333333333e-2, 1.20666667, 1.70333333])
    log_P = np.log(P/P_t)
    # --- numerical solver ---
    log_P = log_P.ravel()
    isna = np.isnan(log_P)
    log_P[isna] = 1.0 
    def func(theta):
        return (a[0]*theta**b[0] + a[1]*theta**b[1] + a[2]*theta**b[2]) / theta - log_P
    theta_0 = np.ones_like(P).ravel()
    theta = root(func, theta_0).x # using scipy.optimize.root
    theta[isna] = np.nan
    theta = theta.reshape(np.shape(P)) if bool(np.shape(P)) else theta.item()
    # ------------------------
    T = theta * T_t - 273.15
    return T








