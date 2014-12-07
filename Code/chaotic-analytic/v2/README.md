chaotic-analytic
================

The algorithm:

1. Generate random analytic function
  1. Generate 2k random real numbers from a Gaussian distribution with mean sqrt(1/2n!) and standard deviation sqrt(1/2n!), for n from 0 to k-1 inclusive, where k is the number of complex coefficients to calculate.
  2. Assign k real numbers to the real components of the Taylor coefficients and the other k real numbers to the imaginary components of the Taylor coefficients.
  3. Create lambda function that performs the Taylor expansion, f(z).
2. Calculate corresponding wavefunction as psi = f(z) exp(-(1/2)(zz\*))
3. Calculate corresponding probability current as **J** = Im{psi\* grad psi} - rho (x**y** - y**x**)
