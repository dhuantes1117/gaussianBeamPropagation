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
from playsound import playsound

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

# space discretization
x_list  = np.linspace(x_l, x_r, N)
dx      = x_list[1] - x_list[0]

# time discretization
N_t = 2000
total_time = 1500 * 5e-5
dt = 1e-4 
t_list  = np.linspace(0, N_t * dt, N_t)
dt      = t_list[1] - t_list[0]


# 2nd FD derivative for Kinetic energy operator p^2 / 2 * m
ID_matrix = np.identity(N)
stencil   = (-(1**2)/(1)) * np.array([[0, 0, 0],[1., -2, 1.], [0, 0, 0]]) / (dx**2)
K_matrix  = convolve2d(ID_matrix, stencil, mode='same')
print(K_matrix)
K_list    = [K_matrix[i][i] for i in range(len(K_matrix))]



# spring constant for harmonic oscillator
k = 0.15
# sets up unperturbed, original potential
V0_list   = x_list * 0
V0_matrix = np.diag(V0_list)
V0_max     = 1 #max(V0_list)

# set up perturbative part of the potential
V1_list    = (0.5) * k * (x_list - ((x_l + x_r) / 2))**2
V1_matrix  = np.diag(V1_list)
V1_max     = max(V1_list)
V1_max     = 1

V_list = V0_list + V1_list
V_matrix = V0_matrix + V1_matrix

# setting up original and perturbative Hamiltonian from kinetic and potential matrices
H0_matrix = K_matrix + V0_matrix
H1_matrix = V1_matrix

# original eigenvalues and vectors
E0_n, eigenvect0rs = np.linalg.eigh(H0_matrix)
norm = np.sqrt(np.sum(V0_list**2))

#E1_n = np.dot(eigenvect0rs, np.matmul(H1_matrix, eigenvect0rs)) # inner product somehow??
E1_n = np.array([np.vdot(eigenvect0rs[:, i], np.matmul(H1_matrix, eigenvect0rs[:, i])) for i in range(len(V0_list))])
C0_mn = np.identity(N)
C1_mn = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if(i == j):
            continue
        Delta_E     = E0_n[i] - E0_n[j]
        #C1_mn[i][j]  = np.vdot(eigenvect0rs[:, j], np.matmul(H1_matrix, eigenvect0rs[:, i])) / Delta_E
        C1_mn[j][i]  = np.vdot(eigenvect0rs[:, j], np.matmul(H1_matrix, eigenvect0rs[:, i]))# / Delta_E
print(C1_mn)


# asdf
E_n = E0_n + E1_n
C_mn = C0_mn + C1_mn
#eigenvect1rs = np.matmul(eigenvect0rs, C_mn) 
eigenvect1rs = np.zeros(np.shape(eigenvect0rs))
for i in range(N):
    eigenvect1rs[:, i] = np.matmul(eigenvect0rs[:, i], C_mn)

"""
E_n = E0_n
C_mn = C0_mn
eigenvect1rs = eigenvect0rs
"""
for n in range(N):
    eigenvect1rs[:, n] = eigenvect1rs[:, n] / np.sqrt(np.sum(eigenvect1rs[:, n]**2))

for n in range(5):
    print(np.sum(eigenvect0rs[:, n] * eigenvect0rs[:, n]))
    print(np.sum(eigenvect1rs[:, n] * eigenvect1rs[:, n]))
    plt.plot(x_list, eigenvect0rs[:, n] * eigenvect0rs[:, n])
    plt.plot(x_list, eigenvect1rs[:, n] * eigenvect1rs[:, n])
    plt.show()
    plt.close()

eigenvectors = np.sqrt(eigenvect0rs**2 + eigenvect1rs**2)
eigenvectors = eigenvect0rs + eigenvect1rs

for n in range(N):
    eigenvectors[:, n] = eigenvectors[:, n] / np.sqrt(np.sum(eigenvectors[:, n]**2))

for n in range(5):
    plt.plot(x_list, eigenvectors[:, n] * eigenvectors[:, n])
    plt.show()
    plt.close()

# setting up physical constants for gaussian wave packet so that it fits within
# an arbitrary window size
FullWidth = (x_r - x_l) / 15        # FWHM will be 1/3 of window
x_c = x_l + (x_r - x_l) / 10         # potential will be centered on the left third of the window
x_c = x_l + (x_r + x_l) / 6
a = -4 * np.log(0.5) / FullWidth**2 # proportionality constant

# initializing and normalizing wavefunction
wavefunc = np.exp(-a * (x_list - x_c)**2)

# finding coefficients in energy representation
a_n_0 = [np.dot(eigenvectors[:,i], wavefunc) for i in range(N)]
a_n_0 = np.array([np.vdot(eigenvectors[:,i], wavefunc) for i in range(N)])
a_n_0 = a_n_0 / np.sqrt(np.vdot(a_n_0, a_n_0))
initial_state = np.sum(a_n_0 * eigenvectors, 1)
plt.plot(x_list, initial_state)
plt.show()
plt.close()
init_max = max(initial_state)
print("mod sq a's")
print(np.vdot(a_n_0, a_n_0))


# ------------------------------------------------------------------ PLOTTING AND OUTPUT -------------------------------------------------------------------------------------

fig, ax = plt.subplots(5, 1)
ax[0].set_xticks([])
ax[1].set_xticks([])
ax[2].set_xticks([])

p  = ax[0].plot([], [], 'C1', animated=True)[0]
p0 = ax[0].plot([], [], 'C0', animated=True)[0]           
p1 = ax[1].plot([], [], 'C0', animated=True)[0]           
p2 = ax[2].plot([], [], 'C0', animated=True)[0]           
p3 = ax[3].plot([], [], 'C0', animated=True)[0]           
p4 = ax[4].plot([], [], 'C1', animated=True)[0]           


# ------------------------------------------------------------------ CALC INITAL STATES  -------------------------------------------------------------------------------------
# Initial just to get autoscaling right
t = t_list[3 * len(t_list) // 7]
# Energy rep coefficients
a_n = np.exp(-1j * E_n * t) * a_n_0
# Getting position rep and plotting
state = np.sum(a_n * eigenvectors, 1)
plottable = np.abs(state)
plottable = plottable / np.sqrt(np.vdot(plottable, plottable))
# FT to get momentum amplitudes
FT = np.abs(fft(state))
freq = npft.fftfreq(x_list.shape[-1])
freq = npft.fftshift(freq)
FT   = npft.fftshift(FT)
#setting plot data of matplotlib2DLines objects for animation
p.set_data(x_list, V_list / (V1_max + V0_max))
p0.set_data(x_list, initial_state)
p1.set_data(x_list, state.real)
p2.set_data(x_list, state.imag)
p3.set_data(x_list, plottable)
p4.set_data(freq, FT)

[ax[i].relim() for i in range(len(ax))]
[ax[i].autoscale_view(False, True, False) for i in range(1, len(ax))]


# ------------------------------------------------------------------ CALCULATE STATES @ ALL TIMES  -------------------------------------------------------------------------------------
Potential   = []
InitialState= []
Real        = []
Imag        = []
Mod_sq      = []
Momentum    = []

# Precalculate all functions of interest to save in lookup table (here just to
# animate them smoother)
for t in t_list:
    # Energy rep coefficients
    a_n = np.exp(-1j * E_n * t) * a_n_0
    # Getting position rep and plotting
    state = np.sum(a_n * eigenvectors, 1)
    plottable = np.abs(state)
    plottable = plottable / np.sqrt(np.vdot(plottable, plottable))
    # FT to get momentum amplitudes
    FT = np.abs(fft(state))
    FT   = npft.fftshift(FT)
    #setting plot data of matplotlib2DLines objects for animation
    Potential.append(V_list / (V1_max + V0_max))
    InitialState.append(initial_state / init_max)
    Real.append(state.real)
    Imag.append(state.imag)
    Mod_sq.append(plottable)
    Momentum.append(FT)

# -------------------------------------------------------------------------- Plotting ----------------------------------------------------------

def lookupAtFrame(frame):
    p. set_data(x_list, Potential[frame])
    p0.set_data(x_list, InitialState[frame])
    p1.set_data(x_list, Real[frame])
    p2.set_data(x_list, Imag[frame])
    p3.set_data(x_list, Mod_sq[frame])
    p4.set_data(freq, Momentum[frame])
    return (p, p0, p1, p2, p3, p4)
    

def outputFormat():
    ax[0].set_xlabel("x")
    ax[1].set_xlabel("x")
    ax[2].set_xlabel("x")
    ax[3].set_xlabel("x")
    ax[4].set_xlabel("p")
    p.set_data ([], [])
    p0.set_data([], [])
    p1.set_data([], [])
    p2.set_data([], [])
    p3.set_data([], [])
    p4.set_data([], [])
    return ax 


#plt.plot(x_list, a_n_o)
alpha = 25 / (5e-5)
#animation_interval = int(dt * alpha * (dt / (5e-5))) // 3
animation_interval = int(4e-5 * alpha * (4e-5 / (5e-5))) // 3

plt.tight_layout()
DisplayState  = animation.FuncAnimation(fig, lookupAtFrame, interval=animation_interval, repeat_delay=100, blit=True, frames=len(t_list), save_count = 60 * 3)

#y_lims = [(0, 1.2), (-0.35, 0.35), (-0.35, 0.35), (-0.05, 0.35), (-1, 12)]
#y_lims = [(0, 1.2), (-35, 35), (-35, 35), (-0.05, 0.35), (-1, 300)]
#[ax[i].set_ylim(lim) for i, lim in enumerate(y_lims)]
ax[0].set_ylim((0, 1.2))

#writervideo = animation.FFMpegWriter(fps=60)
#DisplayState.save("EvolvingPacket.mp4", writer=writervideo)

#playsound("Ding.wav")
plt.show()
