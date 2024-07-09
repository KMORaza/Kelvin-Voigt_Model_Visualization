import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
E = 100.0  
eta = 50.0 
t_input = np.linspace(0, 10, 1000)  
epsilon_input = np.sin(t_input)  
def kelvin_voigt(y, t):
    epsilon, sigma = y
    d_epsilon_dt = np.interp(t, t_input, epsilon_input)  
    d_sigma_dt = E * d_epsilon_dt + eta * (d_epsilon_dt - (d_epsilon_dt - epsilon))
    return [d_epsilon_dt, d_sigma_dt]
y0 = [0.0, 0.0]  
t = np.linspace(0, 10, 1000)  
sol = odeint(kelvin_voigt, y0, t)
epsilon = sol[:, 0]
sigma = sol[:, 1]
plt.figure(figsize=(10, 6))
plt.plot(t, epsilon, label='Strain')
plt.plot(t, sigma, label='Stress')
plt.title('Kelvin-Voigt Viscoelastic Model')
plt.xlabel('Time')
plt.ylabel('Strain / Stress')
plt.legend()
plt.grid(True)
plt.show()
