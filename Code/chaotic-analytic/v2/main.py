from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import chaotic_analytic as ca
import plotting as pl

beg = -1
end = 1
delta = 0.01
trunc_index = 178  # 1/factorial(178) is zero


def main():
    """ The main function loop. """

    print "Generating chaotic analytic function..."
    func_variants = ca.generate_function(trunc_index=trunc_index)
    print "...done\n"

    (function, conj_func, x_deriv_func, y_deriv_func) = func_variants

    X = np.arange(beg, end, delta)
    Y = np.arange(beg, end, delta)
    grid_x, grid_y = np.meshgrid(X, Y)

    print "Generating value matrices associated with the function..."
    value_variants = ca.generate_values(X, Y, func_variants)
    print "...done\n"

    (func_values, conj_values, x_deriv_values, y_deriv_values) = \
        value_variants

    print "Adjusted function values for isotropy..."
    isotropic_values = ca.adjust_function(grid_x, grid_y, func_values)
    print "...done\n"

    print "Calculating probability current..."
    prob_current = ca.calculate_current(grid_x, grid_y, value_variants)
    print "...done\n"

    print "Plotting isotropic stationary function as contour plot..."
    pl.contour_plot(grid_x, grid_y, isotropic_values,
                    title=r"Isotropic chaotic function $H(z) = e^{-zz^*}f(z)$, " + str(trunc_index) + " terms",
                    fname="isotropic-contour-{}-terms-{}-to-{}".format(trunc_index, beg, end),
                    beg=beg, end=end)
    print "...done\n"

    print "Plotting probability current as quiver plot..."
    pl.quiver_plot(grid_x, grid_y, prob_current,
                   title=r"Probability Current, $\mathbf{J}_c$, " + str(trunc_index) + " terms",
                   fname="prob-current-quiver-{}-terms-{}-to-{}".format(trunc_index, beg, end),
                   beg=beg, end=end)
    print "...done\n"

    print "Plotting probability current as stream plot..."
    pl.stream_plot(grid_x, grid_y, prob_current,
                   title=r"Probability Current, $\mathbf{J}_c$, " + str(trunc_index) + " terms",
                   fname="prob-current-stream-{}-terms-{}-to-{}".format(trunc_index, beg, end),
                   beg=beg, end=end)
    print "...done\n"

    plt.show()

    return True


if __name__ == "__main__":
    main()
