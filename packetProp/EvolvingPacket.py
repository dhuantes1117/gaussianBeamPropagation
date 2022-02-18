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
N   = 600
x_l = 0
x_r = 5

# setting up spacial
x_list  = np.linspace(x_l, x_r, N)
dx      = x_list[1] - x_list[0]

# time discretization
N_t = 1500
total_time = 1500 * 5e-5
dt = 4e-5 
t_list  = np.linspace(0, N_t * dt, N_t)
dt      = t_list[1] - t_list[0]


# this process initializes the kinetic energy matrix via a convolution as
# opposed to a for loop setting values along the diagonal
ID_matrix = np.identity(N)
stencil   = (-(1**2)/(1)) * np.array([[0, 0, 0],[1., -2, 1.], [0, 0, 0]]) / (dx**2)
K_matrix  = convolve2d(ID_matrix, stencil, mode='same')
print(K_matrix)
K_list    = [K_matrix[i][i] for i in range(len(K_matrix))]



# spring constant for harmonic oscillator
k = 8e3
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
FullWidth = (x_r - x_l) / 10        # FWHM will be 1/3 of window
x_c = x_l + (x_r - x_l) / 3         # potential will be centered on the left third of the window
x_c = (x_r + x_l) / 3
a = -4 * np.log(0.5) / FullWidth**2 # proportionality constant

# initializing and normalizing wavefunction
wavefunc = np.exp(-a * (x_list - x_c)**2)
wavefunc = eigenvectors[:,0] + 2 * eigenvectors[:,1] + 3j * eigenvectors[:,4]
wavefunc = wavefunc / np.sqrt(np.vdot(wavefunc, wavefunc))
wavefunc = np.pad(wavefunc, (200, 0), mode='constant')[:-200]
wavefunc = np.roll(wavefunc, 200)
wavefunc = np.exp(-a * (x_list - x_c)**2)

print("mod sq ksi's")
print(np.vdot(wavefunc, wavefunc))


# finding coefficients in energy representation
#a_n_o = [np.dot(wavefunc, eigenvectors[:,i]) + 0j for i in range(N)]
#a_n = [dx * np.dot(wavefunc, eigenvectors[:,i]) for i in range(N)]
a_n_0 = [np.dot(eigenvectors[:,i], wavefunc) for i in range(N)]
a_n_0 = np.array([np.vdot(eigenvectors[:,i], wavefunc) for i in range(N)])
a_n_0 = a_n_0 / np.sqrt(np.vdot(a_n_0, a_n_0))
print("mod sq a's")
print(np.vdot(a_n_0, a_n_0))

fig, ax = plt.subplots(5, 1)


ax[0].plot(x_list, np.sum(a_n_0 * eigenvectors, 1))

ims0 = []
ims1 = []
ims2 = []
ims3 = []
ims4 = []
for t in t_list:
    a_n = np.exp(-1j * eigenvalues * t) * a_n_0
    state = np.sum(a_n * eigenvectors, 1)
    plottable = np.abs(state)
    plottable = plottable / np.sqrt(np.vdot(plottable, plottable))
    ims3.append(ax[1].plot(x_list, state.real, 'b'))
    ims4.append(ax[2].plot(x_list, state.imag, 'b'))
    ims1.append(ax[3].plot(x_list, plottable, 'b'))
    ims0.append(ax[0].plot(x_list, V_list, 'orange'))
    ims0.append(ax[0].plot(x_list, np.sum(a_n_0 * eigenvectors, 1), 'blue'))
    #phi_p = (1 / np.sqrt(2 * pi)) * np.sum(np.exp(1j*p*x_list) * wavefunc) * dx
    FT = np.abs(fft(state))
    freq = npft.fftfreq(x_list.shape[-1])
    freq = npft.fftshift(freq)
    FT   = npft.fftshift(FT)
    ims2.append(ax[4].plot(freq, FT, 'orange'))

#plt.plot(x_list, a_n_o)
alpha = 25 / (5e-5)
animation_interval = int(dt * alpha * (dt / (5e-5))) // 3
im_ani0  = animation.ArtistAnimation(fig, ims0, interval=animation_interval, repeat_delay=100, blit=True)
im_ani1  = animation.ArtistAnimation(fig, ims1, interval=animation_interval, repeat_delay=100, blit=True)
im_ani2 = animation.ArtistAnimation(fig, ims2,    interval=animation_interval, repeat_delay=100, blit=True)
im_ani3 = animation.ArtistAnimation(fig, ims3,    interval=animation_interval, repeat_delay=100, blit=True)
im_ani4 = animation.ArtistAnimation(fig, ims4,    interval=animation_interval, repeat_delay=100, blit=True)
#writervideo = animation.FFMpegWriter(fps=60)
#im_ani.save("EvolvingPacket.mp4", writer=writervideo)
plt.show()

exit()



