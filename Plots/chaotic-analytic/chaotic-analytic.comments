Comments on chaotic-analytic.py
===============================

I need to write down what this program does and how, for future reference, otherwise I'll come back in a couple of weeks time and have absolutely no idea what I've done.

Imports
-------

* **Line 6:** I've imported division from future so that division is, by default, always float.
* **Line 18:** This `diff` is a SymPy function to do total and partial differentiation. I use it to try and calculate the gradient of the phase (or equivalent of the log of the modulus due to the analyticity).

Constants
---------

* **Lines 20-23:** I put these in because originally my equation for the probability current had them all in as factors. However, I'm clearly choosing these values, so in the end I didn't bother actually putting them in the equation. I've still got them here for future reference, so I know that it's a purposeful choice and not a mistake. The reason I chose these is that everything turns out to be 1, including (importantly) the coefficient in the exponent, exp(-zz\*), which John told me is what I want for the chaotic analytic function.

Plotting
--------

* **Lines 25-72:** These are all of the plotting functions. They're fairly self-explanatory, but perhaps the only ones that need explaining are:
* **Lines 53-54:** The `lw` in `stream_plot` is a simple calculation to get the magnitude of the probability current at each point in the x,y grid so that it can be used to alter the width of the lines in the plot. The `mag` calculation is just finding the norm through conventional sqrt(x^2 + y^2) sort of calculation, whilst the second arbitrarily scales it with the maximum so that it looks good when used to thicken the stream lines.
* **Line 69:** The `[::5]` in the `plt.quiver` plot is simply because plotting all the arrows gets overcrowded and looks bad, whereas only plotting every fifth element of the dataset (and thereby only every fifth arrow) looks much more aesthetically pleasing.
* **Line 70:** Also, I've commented out the `scale` keyword argument because even though sometimes it works really well, sometimes is just messes up the whole plot and makes all the arrows too long, regardless of what actual value you put as the scale factor. Sometimes it makes it looks nicer, though, so I left it commented for future reference.

Generating Chaotic Analytic Function
------------------------------------

* **Line 78:** So John never actually told me what value to put for the standard deviation, but it seems obvious that it should go with the same square root of 1/(2n!) that the mean goes, otherwise it doesn't converge fast enough. I should probably ask him, though.
* **Line 80:** I have to truncate the Taylor expansion somewhere, because I can't continue on to infinity (for computational reasons). Because my mean square is going with the sqrt(1/(2n!)), I can truncate pretty soon. 10 is a reasonable value, suggested by John, which I have used here. Note that the n=10 term *isn't* included, because the `range(trunc_index)` goes from 0 to 9.
* **Lines 82-85:** I have to calculate the real and imaginary parts of the coefficients for each separately, such that their combined mean square is sqrt(1/n!). Thus I choose random numbers from a normal distribution with mean sqrt(1/2!).
* **Line 89-90:** It seemed easiest to me to have the actual chaotic analytic function as a lambda function. That way it can be generated randomly and passed around as a variable between functions.

Calculating the Probability Current
-----------------------------------

This is where things get a bit more complicated, and where I'm making some mistakes that are causing the results to be wrong. I haven't found them yet, but I think it's probably something to do with either calculating the x and y components or finding the derivative of the log of the modulus of the analytic functino (or alternately the derivative of the phase).

* **Line 100-101:** If we say that f(x,y) = Fe^(i phi), then the probability current has a factor of F^2 in both the x and y components. Thus, we need an array containing the modulus of the analytic function, F, at each point (x,y). That's what this array `func_mod` is.
* **Line 104:** I've commented it out now because it wasn't working (I don't know why), but this is the natural way to calculate the factor I need for the probability current. It's quite a tricky problem actually, and I suspect that even though it doesn't complain, this method I'm using still isn't really working. So essentially I need grad(phi), where phi is the phase, which means I need the partial derivatives of phi with respect to x and y (w.r.t. z is identically null). The best way I could think of doing this in my poor state of mind is to make a lambda function that calculates the phase for given (x,y), and using SymPy to partially differentiate it with respect to x and y in separate methods (see Lines 128-145). As I said, it ostensibly works, but I should really check this properly to make sure I'm getting sensible values. It certainly seems a bit dodgy to me.
* **Line 105:** Because log(f(x,y)) is analytic (exclusing f(x,y) is null and on negative real axis), it satisifes the Cauchy Riemann equations and thus partial_x of phi = - partial_y of F and partial_y of phi = partial_x of F. Thus the grad(phi) in the equations for the probability current components can be written instead as z cross grad(log(F)), which is what I've done here. This lambda function simply calculates log(F) for a particular value of x and y, to be differentiated later.
* **Lines 112-122:** This is where the computer intensive calculations happen. I basically plug in the equation for the probability current, with the partial derivative of either the log of the modulus or the phase being the only dodgy bit. The rest is definitely fine.

Calculating the Partial Derivatives
-----------------------------------

These two functions `partial_x` and `partial_y` are basically wrappers for `sympy.mpmath.diff`, made neater so that I can apply it to entire NumPy arrays in one method call. It's by far the slowest part of the program, because the list comprehension I'm using is sluggish. There's not really much I can do about this, though, as SymPy isn't designed to play well with NumPy, so `diff` (which passes arguments to the function given to it) can only be run with floats as arguments, *not* NumPy arrays as I would have hoped. Ah well.

* **Lines 132-133:** The important thing to know about `diff` is that the first argument is the function to differentiate, the second argument is a tuple containing the values to plug in once you've differentiated and the third argument is a tuple determining how many times you differentiate w.r.t. each variable. So (1,0) means differentiate once w.r.t. x and none w.r.t. y. (0,1) means differentiate no times w.r.t. x and once w.r.t. y. (2,3) means differentiate w.r.t. x twice, then w.r.t. y three times, etc. Another important point is that because I pass `grid_x` and `grid_y` to the function, I need to write `xvals[:][0]` instead of just `xvals`. You see, `grid_x` is actually of shape `(120,120)`. Don't ask me why because I'm not sure, but because the grid is square, the matrix is symmetric, so it doesn't matter whether I write `[:][0]` or `[0][:]`. I don't know why I didn't write xvals[:,0] instead. Maybe I should try that out another time.
* **Line 135:** The important two points in this line is that:
    1. I need to cast it to a NumPy array instead of a standard Python in-built array, so that back in `calculate_current` I can multiply the arrays together and gets another array. You can't do that with Python lists, and you particularly can't exponentiate them.
    2. You need to tell NumPy to store the array as floats, because by default SymPy stores them as a special SymPy version of floats, called `mpf` (meaning a real float that can also be -inf or inf). This causes problems later on down the line if you don't explicitly cast it as a bog standard float (which of course is actually a NumPy 64-bit float).
* **Lines 138-145:** This is pretty much identical to `partial_x`, only the (0,1) tuple given to `diff` means that you differentiate w.r.t. y once and w.r.t. x no times, instead of the opposite.

Main program method
-------------------

I find it clearer to wrap all of the main code in a `main()` function> Kind of C-esque, I know, but when you've got command line interfaces but also want to be able to import `main()` in other Python programs, it turns out to be the best option.

* **Line 152:** This saves the lambda function for the chaotic analytic function as `anal_function` to be passed around the place later by various functions. Could I have thought of a more appropriate function name? Yes. Am I going to change it? No.
* **Lines 155-160:** This bit is important, and has been the cause of much confusion and headache. `X` and `Y` are simply 1D NumPy arrays of shape (120). However, when running `meshgrid()`, `grid_x` and `grid_y` become grid variables, i.e. matrices of shape (120, 120). Where do these extra values come from? I'm not sure. Why does this allow the probability current, surface plot and vector plots to work when previously they didn't passing just `X` and `Y` in? I really don't know. That's definitely something to investigate, prefereably by looking through the NumPy and Matplotlib source code to find how `meshgrid()` changes these and how `surface_plot` and others interact with the input values `x` and `y`. But not when I have this deadline.
* **Lines 163-164:** This is just calculating a matrix of values for the chaotic analytic function, so that I can plot it. It's completely superfluous to the calculations of the probability currents, but is good for direct comparisons between the two. Particularly, the dark spots should have currents circulating around them, a tantalising phenomenon which I have so far not been able to get my calculated currents to exhibit. Once again, list comprehension = a little slow, but not much I can do.
* **Line 167:** Once again, this is unnecessary for the probability current calculations, and is merely so that I can plot the chaotic analytic function without it increasing loads at the edges. I'm not entirely certain about the details of why I need to multiply f(x,y) by this to get something respectable to plot, but according to John and John's paper, I do. I actually redo this calculation as part of the probability current method, so this is directly related to what I'm plotting with the current field.
* **Lines 174-178:** Just plotting these adjusted values for the C.A.F. (Chaotic Analytic Function) so that I can visualise them for the purposes already mentioned. Predominantly, the whirling anticlockwise around local minima is an important quality to look for (although note that the range of J won't be the same as of f(x,y). Also note that only if you plot at the length scales of  about 3 does this look good. Any more and it's all concentrated in one small bit. Not sure why this is exactly, or whether this is correct or incorrect. Maybe something else to confirm with John.
* **Line 182:** This is calling the method to calculate the probability current for our generated C.A.F. Ignore the next two lines, they're trying out known solutions to see whether they looks like they should do.
* **Line 188:** This shouldn't cause problems, but often does. Possibly because I still haven't quite got it all down.
* **Line 192:** The stream plot consistently brings about a plethora of different problems, where as soon as I solve one problem, another even uglier and less helpful one appears.

Things to Ask John
------------------

* What should the standard deviation of the Gaussian distribution be when randomly generating the coefficients?
* How do you calculate the gradient of the phase (or equivalently of the modulus)? Is there a simple way I've missed?
* Does my plot for the C.A.F. look right? Is it supposed to disappear for large x or y?
* Why doesn't my probability current look like it actually corresponds to the C.A.F.? Problem with the code? Something I've overcomplicated?
