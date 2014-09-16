# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
import numexpr
from astropy import log as logger

# Project
# ...

__all__ = ['generate_data']

def generate_data(N, v_func, v_func_kwargs=dict(), sigma=5., censor="Q > 5*sigma"):
    """ Generate fake measurements of $Q = v \sin i$ (e.g., stellar rotation, or
        radial velocity of a binary component). `sigma` controls the measurement
        uncertainty (in units of km/s), `v_func` specifies the function to draw
        true velocities from, and `censor` specifies an expression to censor the
        data with.

        Parameters
        ----------
        N : int
            Number of data points to generate.
        v_func : function
            Function to sample from to get true velocities. For example,
            `np.random.lognormal`.
        v_func_kwargs : dict (optional)
            Any kwargs that need to be passed to the function to sample the
            true velicities. For example, for `lognormal()`, need to specify
            `mean` and `sigma`. Size is added automatically by `N`.
        sigma : float, numeric (optional)
            Observational uncertainty in the observed $v \sin i$ in km/s. Default
            is 5 km/s.
        censor : str (optional)
            An expression (as a string) that specifies how to censor the data.
            Default is to only accept velocitie measured >5 times the velocity
            uncertainty.

    """

    v_func_kwargs['size'] = N
    v = v_func(**v_func_kwargs)

    # v = np.random.lognormal(5, sigma=0.01, size=N)
    # v = np.random.lognormal(5, sigma=0.25, size=N)
    sini = np.sin(np.arccos(np.random.uniform(0.,1.,size=N)))
    Q = v*sini

    if isinstance(sigma, float):
        sigma = np.ones(N)*sigma
    else:
        sigma = np.array(sigma)

    if sigma.shape != Q.shape:
        raise ValueError("Shape of sigma is invalid. Must be a scalar, or length N")

    Q = Q + np.random.normal(0., sigma)

    mask = numexpr.evaluate(censor)  # censor the data
    logger.debug("Fraction of censored objects: {}".format((~mask).sum() / float(N)))

    return v[mask], Q[mask], sigma[mask]
