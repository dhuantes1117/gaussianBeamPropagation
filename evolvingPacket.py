#! /bin/python3
import pint
import numpy as np
from scipy.signal import convolve2d
from scipy.constants import h, c, m_e, hbar, e, pi, epsilon_0
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from numpy.fft import fft
import numpy.fft as npft
import scipy.fftpack

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# setting up physical quantities in pint
_h          = Q_(h, 'J s')
_hbar       = Q_(hbar, 'J s')
_c          = Q_(c, 'm s^-1')
_m_e        = Q_(m_e, 'kg')
_e          = Q_(e, 'C')
_epsilon_0  = Q_(epsilon_0, 'C^2 m^-2 N^-1')

# setting up gradation and bounds for the area of interest
N   = 1000
x_l = 0
x_r = 5

# setting up spacial
x_list  = np.linspace(x_l, x_r, N)
dx      = x_list[1] - x_list[0]

# time discretization
N_t = 1000
dt = 5e-5 
t_list  = np.linspace(0, N_t * dt, N_t)
dt      = t_list[1] - t_list[0]


# this process initializes the kinetic energy matrix via a convolution as
# opposed to a for loop setting values along the diagonal
ID_matrix = np.identity(N)
stencil   = (-(1**2)/(2 * 1)) * np.array([[0, 0, 0],[1., -2, 1.], [0, 0, 0]]) / (dx**2)
K_matrix  = convolve2d(ID_matrix, stencil, mode='same')
print(K_matrix)
K_list    = [K_matrix[i][i] for i in range(len(K_matrix))]



# spring constant for harmonic oscillator
k = 5e4
# sets up a diagonal matrix representing a harmonic oscillator potential
# centered between the bounds
V_list   = (0.5) * k * (x_list - ((x_l + x_r) / 2))**2
V_matrix = np.diag(V_list)

# setting up Hamiltonian from kinetic and potential matrices
H_matrix = K_matrix + V_matrix
print(K_matrix[0][0])
print(V_matrix[0][0])
eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
norm = np.sqrt(np.sum(V_list**2))

# setting up physical constants for gaussian wave packet so that it fits within
# an arbitrary window size
FullWidth = (x_r - x_l) / 3        # FWHM will be 1/3 of window
x_c = x_l + (x_r - x_l) / 10         # potential will be centered on the left third of the window
x_c = (x_r + x_l) / 3
a = -4 * np.log(0.5) / FullWidth**2 # proportionality constant

# initializing and normalizing wavefunction
wavefunc = np.exp(-a * (x_list - x_c)**2)
wavefunc = wavefunc / np.sqrt(np.dot(wavefunc*wavefunc, dx * np.ones(N)))
wavefunc = eigenvectors[:,0];
plt.plot(x_list, wavefunc)
plt.show()
plt.close()
print(np.sum(wavefunc**2))
exit()


# finding coefficients in energy representation
#a_n_o = [np.dot(wavefunc, eigenvectors[:,i]) + 0j for i in range(N)]
#a_n = [dx * np.dot(wavefunc, eigenvectors[:,i]) for i in range(N)]
a_n_0 = [np.dot(eigenvectors[:,i], wavefunc) for i in range(N)]
a_n_0 = np.array([np.dot(eigenvectors[:,i], wavefunc) for i in range(N)])
a_n_0 = a_n_0 / np.sqrt(np.dot(a_n_0, a_n_0 * dx))

fig, ax = plt.subplots(5, 1)

ax[0].plot(x_list, V_list / max(V_list))
ax[0].plot(x_list, wavefunc / max(wavefunc))

ims = []
ims2 = []
ims3 = []
ims4 = []
for t in t_list:
    a_n = np.exp(-1j * eigenvalues * t) * a_n_0
    plottable = np.abs(np.sum(a_n * eigenvectors, 1))
    plottable = plottable / np.sqrt(np.dot(plottable*plottable, dx * np.ones(N)))
    ims3.append(ax[1].plot(x_list, np.sum(a_n * eigenvectors[:], 1).real, 'b'))
    ims4.append(ax[2].plot(x_list, np.sum(a_n * eigenvectors[:], 1).imag, 'b'))
    ims.append(ax[3].plot(x_list, plottable, 'b'))
    FT = np.abs(fft(plottable))
    freq = npft.fftfreq(x_list.shape[-1])
    freq = npft.fftshift(freq)
    FT = npft.fftshift(FT)
    ims2.append(ax[4].plot(freq, FT, 'orange'))

#plt.plot(x_list, a_n_o)
im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=100, blit=True)
im_ani2 = animation.ArtistAnimation(fig, ims2, interval=50, repeat_delay=100, blit=True)
im_ani3 = animation.ArtistAnimation(fig, ims3, interval=50, repeat_delay=100, blit=True)
im_ani4 = animation.ArtistAnimation(fig, ims4, interval=50, repeat_delay=100, blit=True)
#writervideo = animation.FFMpegWriter(fps=60)
#im_ani.save("EvolvingPacket.mp4", writer=writervideo)
plt.show()

exit()

