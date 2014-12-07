from __future__ import division

from math import factorial, sqrt

import numpy as np
import numpy.random as rnd


def generate_function(trunc_index=100):
    """ Generates the chaotic analytic function, and return the tuple
    (function, conjugate function, partial derivative w.r.t. x, partial
    derivative w.r.t. y) """

    coeff_real = np.array([rnd.normal(sqrt(0.5/factorial(n)), sqrt(0.5/factorial(n)))
                           for n in xrange(trunc_index)])
    coeff_imag = np.array([rnd.normal(sqrt(0.5/factorial(n)), sqrt(0.5/factorial(n)))
                           for n in xrange(trunc_index)])

    # Temporary override for testing purposes
    coeff_real = np.array([n for n in xrange(trunc_index)])
    coeff_imag = np.array([n for n in xrange(trunc_index)])

    coeff = coeff_real + coeff_imag*1j
    conj_coeff = coeff_real - coeff_imag*1j

    function = lambda x,y: np.sum(np.array([coeff[i] * (x + y*1j)**i
                                            for i in xrange(trunc_index)]))

    conj_func = lambda x,y: np.sum(np.array([conj_coeff[i] * (x - y*1j)**i
                                             for i in xrange(trunc_index)]))

    x_deriv_func = lambda x,y: np.sum(np.array([i * coeff[i] * (x + y*1j)**(i - 1)
                                                for i in xrange(trunc_index)]))

    y_deriv_func = lambda x,y: np.sum(np.array([1j*i * coeff[i] * (x + y*1j)**(i - 1)
                                                for i in xrange(trunc_index)]))

    return (function, conj_func, x_deriv_func, y_deriv_func)
