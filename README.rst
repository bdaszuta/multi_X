multi_X_SWSH
============
**multi_X_SWSH** is a clean, minimal implementation of the spin-weighted
spherical harmonic transformation for (half) integer spin-weighted fields on
:math:`\mathbb{S}^2` (see :ref:`references` for conventions and
definitions). The complexity scales as :math:`\mathcal{O}(L^3)` where :math:`L`
is the band-limit.

In particular ``multi_X_SWSH.sf_to_salm`` provides:

.. math::
   :nowrap:

   \begin{equation}
      \begin{aligned}
      \mathcal{F}:{}_s f(\vartheta,\,\varphi) \mapsto {}_sa_{lm},
      \end{aligned}
   \end{equation}

together with ``multi_X_SWSH.salm_to_sf`` implementing:

.. math::
   :nowrap:

   \begin{equation}
      \begin{aligned}
      \mathcal{F}^{-1}: {}_sa_{lm} \mapsto {}_s f(\vartheta,\,\varphi).
      \end{aligned}
   \end{equation}


Example usage
-------------
.. code-block:: python

    from numpy import (abs, max)

    import multi_X_SWSH as mXs

    # specify spin-weights and band-limits
    h_s, h_L_th, h_L_ph = -1, 7, 5   # half-integer * 2
    s, L_th, L_ph = 3, 64, 48

    # infer appropriate number of samples based on band-limit
    h_N_th, h_N_ph = mXs.L_to_N(L_th=h_L_th, L_ph=h_L_ph,
                                is_half_integer=True)
    N_th, N_ph = mXs.L_to_N(L_th=L_th, L_ph=L_ph, is_half_integer=False)

    # generate (half)-integer test data
    h_salm = mXs.generate_random_salm(s=h_s, L_th=h_L_th, L_ph=h_L_ph,
                                      is_half_integer=True)

    salm = mXs.generate_random_salm(s=s, L_th=L_th, L_ph=L_ph,
                                    is_half_integer=False)

    # transform to functions
    h_sf = mXs.salm_to_sf(s=h_s, alm=h_salm, N_th=h_N_th, N_ph=h_N_ph,
                          is_half_integer=True)

    sf = mXs.salm_to_sf(s=s, alm=salm, N_th=N_th, N_ph=N_ph,
                        is_half_integer=False)

    # transform back
    rh_salm = mXs.sf_to_salm(s=h_s, f=h_sf, is_half_integer=True)
    r_salm = mXs.sf_to_salm(s=s, f=sf, is_half_integer=False)

    # compute max. abs. error
    h_err = max(abs(rh_salm - h_salm))
    err = max(abs(r_salm - salm))
With the above we find ``h_err = 1.04e-15`` and ``err = 2.45e-14``. See also
:ref:`examples` for a few further use cases and :ref:`function_listing`.


Code technology
---------------
... ``pyfftw``, ``numba``

.. _references:

References
----------

Florian Beyer, Boris Daszuta, Jörg Frauendiener, and Ben Whale.
*Numerical evolutions of fields on the 2-sphere using a spectral method based on
spin-weighted spherical harmonics*. Classical and Quantum Gravity,
31(7):075019, 2014.
`arXiv (1308.4729) <https://arxiv.org/abs/1308.4729>`_

Florian Beyer, Boris Daszuta, and Jörg Frauendiener.  *A spectral method for
half-integer spin fields based on spin-weighted spherical harmonics*. Classical
and Quantum Gravity, 32(17):175013, 2015.
`arXiv (1502.07427) <https://arxiv.org/abs/1502.07427>`_



TODO [in order, roughly]
------------------------
* Clean up eth.
* Clean up docstrings and provide prototype '>>>'
* Expose construction of analytical sYlm & Wigner_d etc.
* The internal _INT_PREC conversion for half-integer should be tidier.
* Joblib can probably be removed in favour of threads + jit nogil.
* Complete tests (together with doctests)
* Compact passed data when using (L_th > L_ph) \/ (L_th_pad != L_th)
* The whole thing could be wrapped into a single, (JIT) compiled function.
* Allow for a T2 without truncation.
