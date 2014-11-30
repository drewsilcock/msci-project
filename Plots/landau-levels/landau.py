from __future__ import division

from math import e
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import hot, gray, hsv
from mpl_toolkits.mplot3d import Axes3D

hbar = 1
normalisation = 1
mass = 1
ang_freq = 1
plotdir = "plots/"


def plot_cont_mag(x, y, cpx_number, fname=False):

    fig = plt.figure()

    cont_mag_plot = plt.contourf(x, y, abs(cpx_number), 100, cmap=hot)

    if fname:
        plt.savefig(plotdir + fname + ".eps")
        print "Figure '" + fname + "' successfully saved."
    else:
        plt.show()

    return True


def plot_surf_mag(x, y, cpx_number, fname=False):

    fig = plt.figure()
    axes = fig.gca(projection='3d')

    surf_mag_plot = axes.plot_surface(x, y, abs(cpx_number), cmap=hot,
                                      linewidth=0, antialiased=True)
    if fname:
        plt.savefig(plotdir + fname + ".eps")
        print "Figure '" + fname + "' successfully saved."
    else:
        plt.show()

    return True


def plot_phase(x, y, cpx_number, fname=False):

    fig = plt.figure()

    phase_plot = plt.contourf(x, y, np.angle(cpx_number), 100, cmap=hsv)

    if fname:
        plt.savefig(plotdir + fname + ".eps")
        print "Figure '" + fname + "' successfully saved."
    else:
        plt.show()

    return True


def plot_stream_vector(x, y, vector, fname=False):

    fig = plt.figure()

    mag = np.sqrt(np.power(abs(vector[0]), 2) + np.power(abs(vector[1]), 2))
    lw = 5*mag/mag.max()

    vect_plot = plt.streamplot(x, y, vector[0], vector[1],
                               linewidth=lw, color='k', density=0.6)

    if fname:
        plt.savefig(plotdir + fname + ".eps")
        print "Figure '" + fname + "' successfully saved."
    else:
        plt.show()

    return True


def plot_quiv_vector(x, y, vector, fname=False):

    fig = plt.figure()

    quiver_plot = plt.quiver(vector[0][::20, ::20], vector[1][::20, ::20],
                             scale=1/0.1)

    if fname:
        plt.savefig(plotdir + fname + ".eps")
        print "Figure '" + fname + "' successfully saved."
    else:
        plt.show()

    return True


def straight_wavefunction(x, y, t):

    psi = normalisation * \
        e**(-0.5 * (mass * ang_freq / hbar) * x**2) * \
        e**(-0.5 * 1j * hbar * ang_freq * t)

    return psi


def circ_wavefunction(x, y, t):

    psi = normalisation * \
        e**(-0.25 * (mass * ang_freq / hbar) * (x**2 + y**2)) * \
        e**(-0.5 * 1j * hbar * ang_freq * t)

    return psi


def straight_current(x, y):

    curr_x = np.zeros(x.shape)
    curr_y = (normalisation**2) * ang_freq * \
        e**(-(mass * ang_freq / hbar) * (x**2)) * x

    return np.array([curr_x, curr_y])


def circ_current(x, y):

    curr_x = 0.5 * ang_freq * (normalisation**2) * y * \
        e**(-0.5 * (mass * ang_freq / hbar) * (x**2 + y**2))

    curr_y = -0.5 * ang_freq * (normalisation**2) * x * \
        e**(-0.5 * (mass * ang_freq / hbar) * (x**2 + y**2))

    return np.array([curr_x, curr_y])


def evolve(psi, delta_t):

    psi_evolved = psi * \
        e**(-0.5 * 1j * hbar * ang_freq * delta_t)

    return psi_evolved


def main():
    # Main function

    delta = 0.025
    x = np.arange(-5.0, 5.0, delta)
    y = np.arange(-5.0, 5.0, delta)
    X, Y = np.meshgrid(x, y)

    print "Calculating wavefunctions..."
    psi_str = straight_wavefunction(X, Y, 0)
    psi_circ = circ_wavefunction(X, Y, 0)
    print "Wavefunctions successfully calculated."

    print "Calculated probability currents..."
    j_str = straight_current(X, Y)
    j_circ = circ_current(X, Y)
    print "Probability currents successfully calculated."

    print "Plotting wavefunctions..."
    plot_cont_mag(X, Y, psi_str, fname="str_psi_cont")
    plot_surf_mag(X, Y, psi_str, fname="str_psi_surf")
    plot_cont_mag(X, Y, psi_circ, fname="circ_psi_cont")
    plot_surf_mag(X, Y, psi_circ, fname="circ_psi_surf")
    print "Wavefunctions successfully plotted."

    print "Plotting probability currents..."
    plot_quiv_vector(X, Y, j_str, fname="str_j_quiv")
    plot_stream_vector(X, Y, j_str, fname="str_j_stream")
    plot_quiv_vector(X, Y, j_circ, fname="circ_j_quiv")
    plot_stream_vector(X, Y, j_circ, fname="circ_j_stream")
    print "Probability currents successfully plotted."

    return True


if __name__ == "__main__":
    main()
