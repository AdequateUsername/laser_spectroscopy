"""fit Gaussians to the four features on the Rb spectrum"""
import numpy as np
from scipy.optimize import curvefit
import matplotlib.pyplot as plt

data = np.loadtxt("data/RB_SPEC.csv", delimiter=',', skiprows=2)
time = data[::, 0][48500:]  # starting data from time of first peak
c1 = data[::, 1][48500:]
c4 = data[::, 2][48500:]


def gaussian_func(x, x0, sigma, A):
    return A * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

initial_guess = [23, 1, 40]
x_fit = np.linspace(20, 25, 10000)
gaussian_fit = curve_fit(gaussian_func, energy, events, p0=initial_guess)
gaussian_y_fit = gaussian_func(x_fit, *gaussian_fit[0])

plt.grid()
plt.plot(time, c1)
plt.plot(time, c4)
plt.xlabel("time (s)")
plt.ylabel("voltage (V)")
plt.show()
