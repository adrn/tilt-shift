# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from astropy import log as logger
import astropy.units as u
from scipy.misc import logsumexp

# Project
# ...

__all__ = ['']

def ln_integrand(x, mu, sigma, i):
    return -0.5*((x-mu*np.sin(i))/sigma)**2 - 0.5*np.log(2*np.pi) \
        - np.log(sigma) + np.log(np.sin(i))

def ln_prior(a_k):
    """ Using the prior

            p(a_k) ∝ 1/a_k
            ln(p) = -ln(a_k)
    """

    if np.any(a_k < 0.):
        return -np.inf

    return -np.sum(np.log(a_k))

def _ln_likelihood_explicit(a_k, v_k, Q, sigma_Q):
    """ Directly turn sums into loops"""

    neyes = 100
    eyes = np.linspace(0., np.pi/2., neyes)  # grid of inclination angles [rad]
    di = eyes[1] - eyes[0]

    l = 0.
    for a,v in zip(a_k,v_k):
        rs = np.zeros_like(eyes)
        for j,eye in enumerate(eyes):
            r = np.exp(ln_integrand(Q,v,sigma_Q,eye))
            rs[j] = r

        l += a * di * rs.sum() / sigma_Q / np.sqrt(2*np.pi)

    ll = np.log(l)

    if np.isnan(ll):
        return -np.inf

    return ll

def ln_likelihood_explicit(a_k, v_k, Qs, sigma_Qs):
    N = len(Qs)
    rs = np.zeros_like(Qs)
    for n in range(N):
        rs[n] = _ln_likelihood_explicit(a_k, v_k, Qs[n], sigma_Qs[n])
    return np.sum(rs)

def ln_likelihood_fast(a_k, v_k, Qs, sigma_Qs):
    """ """

    J = 128  # number of i's to use in dumb-ass integration
    K = len(a_k)  # number of mixture components
    N = len(Qs)  # number of data points
    eyes = np.linspace(0., np.pi/2., J)  # grid of inclination angles [rad]
    di = (eyes[1] - eyes[0])

    # N,K,J
    a_k = a_k[np.newaxis,:,np.newaxis]
    v_k = v_k[np.newaxis,:,np.newaxis]
    eyes = eyes[np.newaxis,np.newaxis,:]
    Qs = Qs[:,np.newaxis,np.newaxis]
    sigma_Qs = sigma_Qs[:,np.newaxis,np.newaxis]

    ln_int = ln_integrand(Qs,v_k,sigma_Qs,eyes).reshape(N, J*K)
    coeff = np.repeat(a_k*di/sigma_Qs/np.sqrt(2*np.pi), J, axis=2)
    coeff = coeff.reshape(ln_int.shape)
    ll = logsumexp(ln_int,
                   b=coeff, axis=1)

    return ll

def ln_posterior(a_k, v_k, Q, sigma_Q):
    lp = ln_prior(a_k)
    if np.any(np.isinf(lp)):
        return -np.inf

    ll = ln_likelihood_fast(a_k, v_k, Q, sigma_Q).sum()
    if np.any(np.isinf(ll)):
        return -np.inf

    return lp + ll
