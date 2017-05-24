What is this?
-------------
A clean, minimal implementation of the SWSH transformation described in [1, 2]
for (half) integer fields providing the nodal <-> modal map:
${}_s f \Longleftrightarrow {}_s a_{lm}$.

References:
[1] Beyer, Florian, Boris Daszuta, and Jörg Frauendiener.
    "A spectral method for half-integer spin fields based on spin-weighted
     spherical harmonics."
    Classical and Quantum Gravity 32, no. 17 (2015): 175013.
[2] Beyer, Florian, Boris Daszuta, Jörg Frauendiener, and Ben Whale.
    "Numerical evolutions of fields on the 2-sphere using a spectral method
    based on spin-weighted spherical harmonics."
    Classical and Quantum Gravity 31, no. 7 (2014): 075019.


What is requried?
-----------------
fastcache, numba, numpy, pyfftw and joblib;

For the careful:
pytest & (non-python inotify for auto_run_test.sh)

For the explatoratory (see examples):
matplotlib

To build docs:
numpydoc & sphinx

How do I use this?
------------------
import multi_SWSH as ms

See also the examples folder.


TODO [in order, roughly]:
-------------------------
(*) Add a trivial Dirac example from the other codebase.
(*) Clean up docstrings and provide prototype '>>>'
(*) Expose construction of analytical sYlm & Wigner_d etc.
(*) The internal _INT_PREC conversion for half-integer should be tidier.
(*) Joblib can probably be removed in favour of threads + jit nogil.
(*) Complete tests (together with doctests)
(*) Compact passed data when using (L_th > L_ph) \/ (L_th_pad != L_th)
(*) The whole thing could be wrapped into a single, (JIT) compiled function.
(*) Allow for a T2 without truncation.
(*) Make this into a fully fledged package and release it... maybe ;)
