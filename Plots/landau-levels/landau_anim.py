from __future__ import division

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.cm import hot, gray


def plot_mag(x, y, psi):

    fig = plt.figure()

    contf = plt.contourf(x, y, abs(psi), 100, cmap=hot)

    plt.show()

    return True


def wavefunction(x, y, t):

    hbar = 1
    normalisation = 1
    mass = 1
    ang_freq = 1

    psi = normalisation*(math.e**(-0.25*(mass*ang_freq/hbar)*(x**2 + y**2)))*\
        math.e**(- 0.5*1j*hbar*ang_freq*t)

    return psi


def update()


def main():
    # Main function

    delta = 0.005
    x = np.arange(-5.0, 5.0, delta)
    y = np.arange(-5.0, 5.0, delta)
    X, Y = np.meshgrid(x, y)

    psi = wavefunction(X, Y, 5)

    plot_mag(X, Y, psi)

    return True


if __name__ == "__main__":
    main()
