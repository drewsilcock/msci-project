from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

plotdir = "plots/"
ext = ".eps"


def contour_plot(x, y, values, title=False, fname=False):
    """ Plots the matrix of (complex) values from the chaotic analytic function
    as contour plot. """

    fig = plt.figure()

    cont_plot = plt.contourf(x, y, abs(values), 100, cmap=cm.hot)

    # plt.tick_params(
    #     axis='both', which='both',
    #     bottom='off', top='off', left='off', right='off',
    #     labelbottom='off', labeltop='off', labelleft='off', labelright='off')

    plt.xlabel("$x$")
    plt.ylabel("$y$")

    if title:
        plt.title(title)

    if fname:
        plt.savefig(plotdir + fname + ext)
        print "Contour plot successfully saved."

    return True


def quiver_plot(x, y, vector, title=False, fname=False, beg=False, end=False):
    """ Plot a vector field using the `quiver` plotting method; looks like
    little arrows pointing, makes for a good intuitive picture of the vector
    field. """

    fig = plt.figure()

    step = 1

    quiver_plot = plt.quiver(x[::step, ::step], y[::step, ::step],
                             abs(vector[0])[::step, ::step],
                             abs(vector[1])[::step, ::step])

    plt.xlabel("$x$")
    plt.ylabel("$y$")

    if beg and end:
        plt.xlim(beg, end)
        plt.ylim(beg, end)

    if title:
        plt.title(title)

    if fname:
        plt.savefig(plotdir + fname + ext)
        print "Contour plot successfully saved."

    return True


def stream_plot(x, y, vector, title=False, fname=False, beg=False, end=False):
    """ Plot a vector field using the `stream` plotting method; more like the
    flow of a river and therefore good for thinking about the divergence and
    curl, but perhaps less intuitively useful. """

    fig = plt.figure()

    mag = np.sqrt(np.power(abs(vector[0]), 2) + np.power(abs(vector[1]), 2))
    lw = 5 * mag/mag.max()

    vect_plot = plt.streamplot(x, y, abs(vector[0]), abs(vector[1]),
                               linewidth=2, color='blue', density=0.6)

    plt.xlabel("$x$")
    plt.ylabel("$y$")

    if beg and end:
        plt.xlim(beg, end)
        plt.ylim(beg, end)

    if title:
        plt.title(title)

    if fname:
        plt.savefig(plotdir + fname + ext)
        print "Contour plot successfully saved."

    return True
