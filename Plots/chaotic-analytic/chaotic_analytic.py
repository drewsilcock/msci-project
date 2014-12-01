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

def generate_function():
    """ Generates the random chaotic analytic function. """

    # TODO: Ask John to make sure the standard deviation is right here

    trunc_index = 10 # The index to truncate to in Taylor expansion

    coeff_real = np.array([rand.normal(sqrt(0.5/factorial(n)), sqrt(0.5/factorial(n)))
                           for n in range(trunc_index)])
    coeff_imag = np.array([rand.normal(sqrt(0.5/factorial(n)), 0.5/factorial(n))
                           for n in range(trunc_index)])

    coeff = coeff_real + coeff_imag*1j

    analytic_func = lambda z: np.sum(np.array([coeff[i] * z**i
                                               for i in range(trunc_index)]))

    return analytic_func


def surface_plot(x, y, values):
    """ Plots the matrix of (complex) values from the chaotic analytic function
    as surface plot. """

    fig = plt.figure()
    axes = fig.gca(projection='3d')

    surf_plot = axes.plot_surface(x, y, abs(values), cmap=hot, rstride=1,
                                  cstride=1, linewidth=0, antialiased=True)

    return True


def contour_plot(x, y, values):
    """ Plots the matrix of (complex) values from the chaotic analytic function
    as contour plot. """

    fig = plt.figure()

    cont_plot = plt.contourf(x, y, abs(values), 100, cmap=hot)

    return True


def calculate_current(x, y, analytic_func):
    """ Calculates the probability current for the circular gauge for the
    Landau problem, using the value of f(x,y) as the given chaotic analytic
    function. """

    func_mod = np.array([[analytic_func(xval + yval*1j) for xval in x]
                          for yval in y])

    print "Made it this far..."
    func_phase = lambda X,Y: cmath.log(analytic_func(X + Y*1j)).imag
    print "Made it past the func_phase declaration"

    current_x = func_mod**2 * \
        np.power(e, (-(x**2 + y**2))) * \
        (partial_x(func_phase, x, y) + y)

    current_y = func_mod**2 * \
        np.power(e, (-(x**2 + y**2))) * \
        (partial_y(func_phase, x, y) - x)

    return np.array([current_x, current_y])


def partial_x(function, xvals, yvals):
    """ Computes the partial derivative of a function of two variables f(x,y)
    with respect to x """

    test = np.array([[x+y for x in xvals[:][0]] for y in yvals[:][0]])

    print test.shape
    print type(test[0][0])

    print function(xvals[0][0], yvals[0][0])

    ret = [[diff(function, (xval, yval), (1, 0)) for xval in xvals[:][0]]
           for yval in yvals[:][0]]

    print ret
    return ret


def partial_y(function, xvals, yvals):
    """ Computes the partial derivative of a function of two variables f(x,y)
    with respect to y """

    ret = [[diff(function, (xval, yval), (0, 1)) for xval in xvals[:][0]]
           for yval in yvals[:][0]]

    return ret


def stream_plot(x, y, vector):

    fig = plt.figure()

    mag = np.sqrt(np.power(abs(vector[0]), 2) + np.power(abs(vector[1]), 2))
    lw = np.array(5 * mag/mag.max(), dtype=float)

    vect_plot = plt.streamplot(x, y, vector[0], vector[1],
                               linewidth=lw, color='k', density=0.6)

    return True


def quiv_plot(x, y, vector):

    fig = plt.figure()

    quiver_plot = plt.quiver(vector[0][::5, ::5], vector[1][::5, ::5])#,
                             #scale=1)

    return True


def main():
    """ Get generated function, construct matrix of values and plot them. """

    print "Generating chaotic analytic function..."
    anal_func = generate_function()
    print "Chaotic analytic function successfully generated."

    start = -3
    end = 3
    delta = 0.05
    X = np.arange(start, end, delta)
    Y = np.arange(start, end, delta)
    grid_x, grid_y = np.meshgrid(X, Y)

    print "Generating value matrix for analytic function..."
    func_values = np.array([[anal_func(x + y*1j) for x in X]
                            for y in Y])
    print "Value matrix for analytic function successfully generated."

    adjusted_values = np.zeros(func_values.shape, dtype='complex')

    print "Adjusting values by exponential..."
    adjusted_values = func_values*e**(-0.5*(grid_x*grid_x + grid_y*grid_y))
    print "Values successfully adjusted by exponential."

    print adjusted_values.shape

    print "Plotting surface of analytic function..."
    surface_plot(grid_x, grid_y, adjusted_values)
    print "Surface of chaotic analytic function successfully plotted."

    print "Plotting contour of analytic function..."
    contour_plot(grid_x, grid_y, adjusted_values)
    print "Contour of chaotic analytic function successfully plotted."

    print "Calculating probability current..."
    prob_current = calculate_current(grid_x, grid_y, anal_func)
    #prob_current = calculate_current(grid_x, grid_y, lambda z: 1)
    print "Probability current calculated."

    print "Plotting probability current..."
    quiv_plot(grid_x, grid_y, prob_current)
    #stream_plot(X, Y, prob_current)
    print "Probability current successfully plotted."

    plt.show()


if __name__ == "__main__":
        main()
