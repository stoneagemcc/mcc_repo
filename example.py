from psychrometrics import get_T, get_w, get_phi, get_h, get_v, get_T_wet, get_T_dew
from psy_chart import plot


### Find any states given any 2 states:
# T     : Dry-bulb Temperature [degC]
# w     : Humidity Ratio or Absolute Humidty [kg-water/kg-dry-air]
# phi   : Relative Humidity []
# h     : Enthalpy of Air-Vapor Mixture [kJ/kg-dry-air]
# T_wet : Wet-bulb Temperature [degC]
# v     : Specific Volume [m3/kg-dry-air]

### Others:
# P     : atmospheric Pressure [kPa]
# T_dew : Dew-point Temperature (given w) [degC]


### you must supply keyword arguments in get_*() function
### Default: P = 101.325 [kPa]



# Single Test Point (T=20, w=0.01, phi=0.685, h=45.48, T_wet=16.25, v=0.8438)
T = 20 # [degC]
phi = 0.685 # no unit
w = get_w(T=T, phi=phi) # [kg-water/kg-dry-air]
print(f'Humidity Ratio (w): {w:.4f} @ Temp. (T): {T:.2f} degC & Rel. Humidity (phi): {phi:.4f}')


w = 0.01 # [kg-water/kg-dry-air]
phi = 0.685 # no unit
T_wet = get_T_wet(w=w, phi=phi) # [degC]
print(f'Wet-buld Temp. (T_wet): {T_wet:.2f} @ Humidity Ratio (w): {w:.4f} & Rel. Humidity (phi): {phi:.4f}')


plot(T=20, w=0.01) # plot psy. chart (point marked with 'X')



# Multiple Ordered Points
import numpy as np
T = np.array([20, 25, 35])
w = np.array([0.01, 0.015, 0.012])

phi = get_phi(T=T, w=w)
plot(T=T, phi=phi)



# Many Points
import matplotlib.pyplot as plt
T = np.linspace(-10, 40, 51)
phi = 0.5

T_wet = get_T_wet(T=T, phi=phi)
plt.plot(T, T_wet, marker='x')
plt.xlabel('Dry-bulb Temp. (T) [degC]')
plt.ylabel('Wet-bulb Temp. (T_wet) [degC]')
plt.title(f'Dry-bulb Temp. (T) vs Wet-bulb Temp. (T_wet) @ Relative Humidity = {phi*100}%')
plt.show()

plot(T=T, phi=phi)




