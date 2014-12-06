""" Profiles the chaotic_analytic.py script to show which bits take up the most
computing power, and what takes the longest time. """

import cProfile

import chaotic_analytic as ca

res = cProfile.run("ca.main()")

with open("results.txt", "w") as f:
    f.write(res)
