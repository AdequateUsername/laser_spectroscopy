"""fit Gaussians to the four features on the Rb spectrum"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

data = np.loadtxt("data/RB_SPEC.csv", delimiter=',', skiprows=2)
time = data[::, 0][48500:]  # starting data from time of first peak
c1 = data[::, 1][48500:]
c4 = data[::, 2][48500:]

c4_1_start = np.where(time == 2.2881933E-02)[0][0]
c4_2_start = np.where(time == 2.8933722E-02)[0][0]
c4_2_end = np.where(time == 3.5196122E-02)[0][0]
c4_3_start = np.where(time == 4.2437645E-02)[0][0]
c4_3_end = np.where(time == 4.7107571E-02)[0][0]
c4_4_start = np.where(time == 5.1983731E-02)[0][0]
c4_4_end = np.where(time == 5.8357030E-02)[0][0]


def gaussian_func(x, x0, sigma, A, offset):
    return -1 * A * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + offset


# Gaussians for channel 4
# peak 1
initial_guess_c4_1 = [0.026, 0.002, 0.05, -0.05]
x_fit_1 = np.linspace(0.023, 0.03, 10000)
gaussian_fit_c4_1 = curve_fit(gaussian_func, time[c4_1_start:c4_2_start], c4[c4_1_start:c4_2_start], p0=initial_guess_c4_1)
gaussian_y_fit_c4_1 = gaussian_func(x_fit_1, *gaussian_fit_c4_1[0])
# peak 2
initial_guess_c4_2 = [0.033, 0.002, 0.05, -0.05]
x_fit_2 = np.linspace(0.028, 0.036, 10000)
gaussian_fit_c4_2 = curve_fit(gaussian_func, time[c4_2_start:c4_2_end], c4[c4_2_start:c4_2_end], p0=initial_guess_c4_2)
gaussian_y_fit_c4_2 = gaussian_func(x_fit_2, *gaussian_fit_c4_2[0])
# peak 3
initial_guess_c4_3 = [0.045, 0.002, 0.05, -0.05]
x_fit_3 = np.linspace(0.041, 0.049, 10000)
gaussian_fit_c4_3 = curve_fit(gaussian_func, time[c4_3_start:c4_3_end], c4[c4_3_start:c4_3_end], p0=initial_guess_c4_3)
gaussian_y_fit_c4_3 = gaussian_func(x_fit_3, *gaussian_fit_c4_3[0])
# peak 4
initial_guess_c4_4 = [0.055, 0.002, 0.05, -0.05]
x_fit_4 = np.linspace(0.052, 0.059, 10000)
gaussian_fit_c4_4 = curve_fit(gaussian_func, time[c4_4_start:c4_4_end], c4[c4_4_start:c4_4_end], p0=initial_guess_c4_4)
gaussian_y_fit_c4_4 = gaussian_func(x_fit_4, *gaussian_fit_c4_4[0])

print("peak widths:")
print(gaussian_fit_c4_1[0][1], " ± ", np.sqrt(gaussian_fit_c4_1[1][1][1]))
print(gaussian_fit_c4_2[0][1], " ± ", np.sqrt(gaussian_fit_c4_2[1][1][1]))
print(gaussian_fit_c4_3[0][1], " ± ", np.sqrt(gaussian_fit_c4_3[1][1][1]))
print(gaussian_fit_c4_4[0][1], " ± ", np.sqrt(gaussian_fit_c4_4[1][1][1]))

plt.grid()
# plot data
# plt.plot(time, c1)
plt.plot(time, c4)
# plt.plot(time[c4_4_start:c4_4_end], c4[c4_4_start:c4_4_end])
# plot channel 4 fits
plt.plot(x_fit_1, gaussian_y_fit_c4_1, color='r')
plt.plot(x_fit_2, gaussian_y_fit_c4_2, color='r')
plt.plot(x_fit_3, gaussian_y_fit_c4_3, color='r')
plt.plot(x_fit_4, gaussian_y_fit_c4_4, color='r')
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("Rubidium Spectrum (no background adjustment)")
plt.savefig("rb_spectrum_no_background_adjustment", dpi=400)
plt.show()
