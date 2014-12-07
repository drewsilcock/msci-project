from __future__ import division

import chaotic_analytic as ca


def main():

    (function, conj_func, x_deriv_func, y_deriv_func) = \
        ca.generate_function()

    print function(1,1)
    print conj_func(1,1)
    print x_deriv_func(1,1)
    print y_deriv_func(1,1)

    return True


if __name__ == "__main__":
    main()
