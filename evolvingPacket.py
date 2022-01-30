#! /bin/python3
import pint
import numpy as np
from scipy.signal import convolve2d
from scipy.constants import h, c, m_e, hbar, e, pi, epsilon_0
import matplotlib.pyplot as plt

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

_h = Q_(h, 'J s')
_hbar = Q_(hbar, 'J s')
_c = Q_(c, 'm s^-1')
_m_e = Q_(m_e, 'kg')
_e = Q_(e, 'C')
_epsilon_0 = Q_(epsilon_0, 'C^2 m^-2 N^-1')

N = 200
x_l = 0
x_r = 50
x_list = np.linspace(x_l, x_r, N)
dx = x_list[1] - x_list[0]

ID_matrix = np.identity(N)
# convolution stencils are written BACKWARDS from intuition
stencil = (-(1**2)/(2 * 1)) * np.array([[0, 0, 0],[1., -2, 1.], [0, 0, 0]]) / (dx**2)
K_matrix = convolve2d(ID_matrix, stencil, mode='same') * (1. / (2 * dx))
k = 0.0001
V_list = (0.5) * k * (x_list - ((x_l + x_r) / 2))**2
V_matrix = np.diag(V_list)
H_matrix = K_matrix + V_matrix
print("K matrix:")
print(K_matrix)
print("V matrix:")
print(V_matrix)

eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)


print("E_o:")
E0 = eigenvalues[0]
print(eigenvalues[0])
print("E_1:")
E1 = eigenvalues[1]
print(eigenvalues[1])
print("E_2:")
E2 = eigenvalues[2]
print(eigenvalues[2])
print("E_3:")
E3 = eigenvalues[3]
print(eigenvalues[3])
print("E_4:")
E4 = eigenvalues[4]
print(eigenvalues[4])
#print("|E_o>:")
#print(eigenvectors[:,0])

x_plus = np.sqrt(2 * np.array([E0, E1, E2, E3]) / k)
for k, bound in enumerate(x_plus):
    fin_ind = int((bound / (x_r - x_l)) * len(eigenvectors[:,k]))
    I = sum([dx * (eigenvectors[:,0][i]**2) for i in range(fin_ind)])
    prob = 1 - 2 * I
    print("Probability of finding E_%d outside of range: %f" %(k, prob))

norm = np.sqrt(np.sum(V_list**2))


plt.plot(x_list, V_list)
plt.plot(x_list, eigenvectors[:,0] * norm)
plt.show()
plt.close()
exit()

