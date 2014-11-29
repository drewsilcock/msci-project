from __future__ import division

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import hot, gray

hbar = 1
normalisation = 1
mass = 1
ang_freq = 1


def plot_mag(x, y, psi):

    fig = plt.figure()

    mag_plot = plt.contourf(x, y, abs(psi), 100, cmap=hot)

    plt.savefig("landau_level.eps")

    return True


def wavefunction(x, y, t):

    psi = normalisation*(math.e**(-0.25*(mass*ang_freq/hbar)*(x**2 + y**2)))*\
        math.e**(- 0.5*1j*hbar*ang_freq*t)

    return psi


def evolve(psi, delta_t):

    psi_evolved = psi*math.e**(- 0.5*1j*hbar*ang_freq*delta_t)

    return psi_evolved


def main():
    # Main function

    delta = 0.005
    x = np.arange(-5.0, 5.0, delta)
    y = np.arange(-5.0, 5.0, delta)
    X, Y = np.meshgrid(x, y)

    psi = wavefunction(X, Y, 0)

    plot_mag(X, Y, psi)

    return True


if __name__ == "__main__":
    main()

# TODO: Vector plot, phase plot
