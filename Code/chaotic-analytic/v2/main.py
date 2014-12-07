from __future__ import division

import numpy as np

import chaotic_analytic as ca

beg = -3
end = 3
delta = 0.25


def main():

    func_variants = ca.generate_function()

    (function, conj_func, x_deriv_func, y_deriv_func) = func_variants

    X = np.arange(beg, end, delta)
    Y = np.arange(beg, end, delta)
    grid_x, grid_y = np.meshgrid(X, Y)

    value_variants = ca.generate_values(X, Y, func_variants)

    (func_values, conj_values, x_deriv_values, y_deriv_values) = \
        value_variants

    # print func_values[0]
    # print conj_values[0]
    # print x_deriv_values[0]
    # print y_deriv_values[0]

    return True


if __name__ == "__main__":
    main()
