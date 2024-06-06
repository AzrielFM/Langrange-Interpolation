import numpy as np
import matplotlib.pyplot as plt

def lagrange_interpolation(x, y, xp):
    yp = 0
    n = len(x)
    for i in range(n):
        p = 1
        for j in range(n):
            if i != j:
                p *= (xp - x[j]) / (x[i] - x[j])
        yp += p * y[i]
    return yp

def newton_interpolation(x, y, xp):
    def divided_differences(x, y):
        n = len(y)
        coef = np.zeros([n, n])
        coef[:,0] = y
        for j in range(1, n):
            for i in range(n - j):
                coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
        return coef[0, :] 

    def newton_polynomial(coef, x, xp):
        n = len(coef) - 1
        p = coef[n]
        for k in range(1, n + 1):
            p = coef[n - k] + (xp - x[n - k]) * p
        return p

    coef = divided_differences(x, y)
    return newton_polynomial(coef, x, xp)

# Data yang diberikan
x_data = np.array([5, 10, 15, 20, 25, 30, 35, 40])
y_data = np.array([40, 30, 25, 40, 18, 20, 22, 15])

# Rentang x untuk plotting
x_plot = np.linspace(5, 40, 500)

# Interpolasi Lagrange
y_lagrange = [lagrange_interpolation(x_data, y_data, xp) for xp in x_plot]

# Interpolasi Newton
y_newton = [newton_interpolation(x_data, y_data, xp) for xp in x_plot]

# Plot hasil interpolasi
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, 'ro', label='Data Points')
plt.plot(x_plot, y_lagrange, 'b-', label='Lagrange Interpolation')
plt.plot(x_plot, y_newton, 'g--', label='Newton Interpolation')
plt.xlabel('Tegangan (kg/mm²)')
plt.ylabel('Waktu Patah (s)')
plt.legend()
plt.title('Interpolasi Polinomial Lagrange dan Newton')
plt.grid(True)
plt.show()