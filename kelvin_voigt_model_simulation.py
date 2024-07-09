import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

class KelvinVoigtModel:
    def __init__(self, E, eta, dt):
        self.E = E  # Young's modulus (elastic modulus)
        self.eta = eta  # Viscosity coefficient
        self.dt = dt  # Time step

    def step_strain(self, t_max, strain_step):
        time = np.arange(0, t_max, self.dt)
        strain = np.zeros_like(time)
        stress = np.zeros_like(time)

        for i in range(len(time)):
            if i == 0:
                stress[i] = 0.0
            else:
                stress_elastic = self.E * strain[i-1]
                stress_viscous = self.eta * (strain[i-1] - strain[i-2]) / self.dt
                stress[i] = stress_elastic + stress_viscous

            if time[i] <= 1.0:
                strain[i] = strain_step * time[i]

        return time, strain, stress

    def relaxation(self, t_max):
        time = np.arange(0, t_max, self.dt)
        strain = np.zeros_like(time)
        stress = np.zeros_like(time)

        for i in range(len(time)):
            if i == 0:
                stress[i] = 0.0
            else:
                stress[i] = stress[i-1] - (stress[i-1] / (self.E * self.dt))

        return time, strain, stress

    def ramp_loading(self, t_max, ramp_rate):
        time = np.arange(0, t_max, self.dt)
        strain = np.zeros_like(time)
        stress = np.zeros_like(time)

        for i in range(len(time)):
            if i == 0:
                stress[i] = 0.0
            else:
                stress[i] = stress[i-1] + ramp_rate * self.E * self.dt

        strain = stress / self.E

        return time, strain, stress

    def validate_model(self, experimental_time, experimental_stress):
        # Simulate stress response using the model
        simulated_stress = self.E * np.gradient(experimental_time, experimental_stress) + \
                           self.eta * np.gradient(experimental_stress, experimental_time)

        # Calculate metrics
        r2 = r2_score(experimental_stress, simulated_stress)
        mse = mean_squared_error(experimental_stress, simulated_stress)

        return r2, mse, simulated_stress

# Example experimental data
experimental_time = np.linspace(0, 10, 100)
experimental_stress = 1.0 * experimental_time + np.random.normal(scale=0.1, size=len(experimental_time))

# Simulation parameters
E = 1.0  # Young's modulus of the spring (elastic modulus)
eta = 0.5  # Viscosity coefficient (dashpot viscosity)
dt = 0.01  # Time step

# Create Kelvin-Voigt model instance
kv_model = KelvinVoigtModel(E, eta, dt)

# Simulate step strain loading
time_step, strain_step, stress_step = kv_model.step_strain(t_max=10.0, strain_step=1.0)

# Validate model with experimental data
r2, mse, simulated_stress = kv_model.validate_model(experimental_time, experimental_stress)

# Plotting results
plt.figure(figsize=(12, 10))

# Plot experimental data and model prediction
plt.subplot(3, 1, 1)
plt.plot(experimental_time, experimental_stress, label='Experimental Stress')
plt.plot(experimental_time, simulated_stress, label='Simulated Stress', linestyle='--')
plt.title('Experimental Data vs Kelvin-Voigt Model Prediction')
plt.xlabel('Time')
plt.ylabel('Stress')
plt.legend()
plt.grid(True)

# Plot step strain loading
plt.subplot(3, 1, 2)
plt.plot(time_step, strain_step, label='Strain')
plt.plot(time_step, stress_step, label='Stress')
plt.title('Step Strain Loading')
plt.xlabel('Time')
plt.ylabel('Strain / Stress')
plt.legend()
plt.grid(True)

# Print metrics
plt.subplot(3, 1, 3)
plt.text(0.1, 0.8, f'R² = {r2:.2f}', fontsize=12, transform=plt.gca().transAxes)
plt.text(0.1, 0.7, f'MSE = {mse:.2e}', fontsize=12, transform=plt.gca().transAxes)
plt.axis('off')

plt.tight_layout()
plt.show()
