""" Generates a random (or chaotic) analytic function as a Taylor series with
random components provided that it converges, and then calculates the
corresponding probability current for the circular gauge in the landau
problem. """

from __future__ import division

import math
import cmath
from math import factorial, sqrt, e

import numpy.random as rand
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import hot, gray, hsv
from mpl_toolkits.mplot3d import Axes3D

from sympy.mpmath import diff

hbar = 1
normalisation = 1
mass = 1
ang_freq = 2

beg = -3
end = 3
delta = 0.1

plotdir = "plots/"
ext = ".eps"

def surface_plot(x, y, values):
    """ Plots the matrix of (complex) values from the chaotic analytic function
    as surface plot. """

    fig = plt.figure()
    axes = fig.gca(projection='3d')

    surf_plot = axes.plot_surface(x, y, abs(values), cmap=hot, rstride=1,
                                  cstride=1, linewidth=0, antialiased=True)

    return True


def contour_plot(x, y, values, title=False, fname=False):
    """ Plots the matrix of (complex) values from the chaotic analytic function
    as contour plot. """

    fig = plt.figure()

    cont_plot = plt.contourf(x, y, abs(values), 100, cmap=hot)

    plt.tick_params(\
        axis='both', which='both',
        bottom='off', top='off', left='off', right='off',
        labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    plt.xlabel("$x$")
    plt.ylabel("$y$")

    if title:
        plt.title(title)

    if fname:
        plt.savefig(plotdir + fname + ext)
        print "Contour plot successfully saved."

    return True


def stream_plot(x, y, vector):
    """ Plot a vector field using the `stream` plotting method; more like the
    flow of a river and therefore good for thinking about the divergence and
    curl, but perhaps less intuitively useful. """

    fig = plt.figure()

    print vector[0][5,5]
    print vector[1][5,5]
    print vector[0].shape
    print vector[1].shape

    mag = np.sqrt(np.power(abs(vector[0]), 2) + np.power(abs(vector[1]), 2))
    lw = 5 * mag/mag.max()

    vect_plot = plt.streamplot(x, y, abs(vector[0]), abs(vector[1]))#,
                               #linewidth=2, color='k', density=0.6)
    plt.xlim(beg, end)
    plt.ylim(beg, end)

    return True


def quiv_plot(x, y, vector):
    """ Plot a vector field using the `quiver` plotting method; looks like
    little arrows pointing, makes for a good intuitive picture of the vector
    field. """

    fig = plt.figure()

    quiver_plot = plt.quiver(x[::5,::5], y[::5,::5],
                             abs(vector[0][::1, ::1]), abs(vector[1][::1, ::1]))#,
                             #scale=1)
    plt.xlim(beg, end)
    plt.ylim(beg, end)

    return True


def generate_function():
    """ Generates the random chaotic analytic function. """

    trunc_index = 100 # The index to truncate to in Taylor expansion

    coeff_real = np.array([rand.normal(sqrt(0.5/factorial(n)), sqrt(0.5/factorial(n)))
                           for n in xrange(trunc_index)])
    coeff_imag = np.array([rand.normal(sqrt(0.5/factorial(n)), sqrt(0.5/factorial(n)))
                           for n in xrange(trunc_index)])

    coeff = coeff_real + coeff_imag*1j

    print coeff

    analytic_func = lambda z: np.sum(np.array([coeff[i] * z**i
                                               for i in xrange(trunc_index)]))

    print analytic_func(1+1j)
    print analytic_func(100+100j)

    return analytic_func


def calculate_wavefunction(x, y, analytic_func):
    """ Calculates the ground state in circular gauge for Landau problem, given
    the analytic function f(x,y). """

    return analytic_func(x + y*1j) * e**(-0.5 * (x**2 + y**2))


def calculate_current(x, y, analytic_func):
    """ Calculates the probability current for the circular gauge for the
    Landau problem, using the value of f(x,y) as the given chaotic analytic
    function. """

    psi = lambda x,y: analytic_func(x + y*1j) * e**(-0.5 * (x**2 + y**2) )
    psi_values = calculate_wavefunction(x, y, analytic_func)

    print "Calculating the x components of the probability current..."
    current_x = -0.5j*(np.ma.conjugate(psi_values) * partial_x(psi, x, y)
                       + psi_values * np.ma.conjugate(partial_x(psi, x, y))) + y
    print "x components of probability current successfully calculated."

    print "Calculating the y components of the probability current..."
    current_y = -0.5j*(np.ma.conjugate(psi_values) * partial_y(psi, x, y)
                       + psi_values * np.ma.conjugate(partial_y(psi, x, y))) - x
    print "y component of probability current successfully calculated."

    return np.array([current_x, current_y])


def partial_x(function, xvals, yvals):
    """ Computes the partial derivative of a function of two variables f(x,y)
    with respect to x """

    print "Differentiating with respect to x..."

    ret = [[diff(function, (xval, yval), (1, 0)) for xval in xvals[:][0]]
           for yval in yvals[:][0]]

    #ret = [[diff(function, (xval, yval), (1, 0)) for xval in xvals]
    #       for yval in yvals]

    print "Successfully differentiated with respect to x."

    return np.array(ret, dtype=complex)


def partial_y(function, xvals, yvals):
    """ Computes the partial derivative of a function of two variables f(x,y)
    with respect to y """

    print "Differentiating with respect to y..."

    ret = [[diff(function, (xval, yval), (0, 1)) for xval in xvals[:][0]]
           for yval in yvals[:][0]]

    #ret = [[diff(function, (xval, yval), (0, 1)) for xval in xvals]
    #       for yval in yvals]

    print "Successfully differentiated with respect to y."

    return np.array(ret, dtype=complex)


def main():
    """ Get generated function, construct matrix of values and plot them. """

    print "Generating chaotic analytic function..."
    anal_func = generate_function()
    print "Chaotic analytic function successfully generated."

    X = np.arange(beg, end, delta)
    Y = np.arange(beg, end, delta)
    grid_x, grid_y = np.meshgrid(X, Y)

    print "Generating value matrix for analytic function..."
    func_values = np.array([[anal_func(x + y*1j) for x in X]
                            for y in Y])
    print "Value matrix for analytic function successfully generated."

    adjusted_values = np.zeros(func_values.shape, dtype='complex')

    print "Adjusting values by exponential..."
    #adjusted_values = np.power(abs(func_values), 2) * \
    #    e**(-(grid_x*grid_x + grid_y*grid_y))
    for i,x in enumerate(X):
        for j,y in enumerate(Y):
            adjusted_values[i][j] = func_values[i][j] * \
                e**(-0.5*(x*x + y*y))
    print "Values successfully adjusted by exponential."

    print "Plotting surface of analytic function..."
    surface_plot(grid_x, grid_y, adjusted_values)
    print "Surface of chaotic analytic function successfully plotted."

    #print "Plotting surface of isotropic analytic function..."
    #surface_plot(grid_x, grid_y, func_values)
    #print "Surface of chaotic analytic function successfully plotted."

    print "Plotting contour of isotropic analytic function..."
    contour_plot(grid_x, grid_y, adjusted_values,
                 title="Isotropic Stationary Random Function",
                 fname="caf")
    print "Contour of chaotic analytic function successfully plotted."

    print "Calculating probability current..."
    prob_current = calculate_current(grid_x, grid_y, anal_func)
    #prob_current = calculate_current(X, Y, anal_func)
    #prob_current = calculate_current(grid_x, grid_y, lambda z: e**(-z**2))
    #prob_current = calculate_current(grid_x, grid_y, lambda z: 1)
    print "Probability current calculated."

    print "Plotting probability current as quiver plot..."
    quiv_plot(grid_x, grid_y, prob_current)
    print "Probability current successfully plotted as quiver plot."

    print "Plotting probability current as stream plot..."
    stream_plot(grid_x, grid_y, prob_current)
    print "Probability current successfully plotted as stream plot."

    plt.show()


if __name__ == "__main__":
        main()
