# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np

# Project
from .model import ln_integrand

__all__ = ['ln_prior', 'ln_likelihood', 'ln_posterior']

def ln_prior(a_k):
    """ Using the prior

            p(a_k) ∝ 1/a_k
            ln(p) = -ln(a_k)
    """

    if np.any(a_k <= 0.):
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

def ln_likelihood(a_k, v_k, Qs, sigma_Qs):
    N = len(Qs)
    rs = np.zeros_like(Qs)
    for n in range(N):
        rs[n] = _ln_likelihood_explicit(a_k, v_k, Qs[n], sigma_Qs[n])
    return np.sum(rs)

def ln_posterior(a_k, v_k, Q, sigma_Q):
    lp = ln_prior(a_k)
    if np.any(np.isinf(lp)):
        return -np.inf

    ll = ln_likelihood(a_k, v_k, Q, sigma_Q).sum()
    if np.any(np.isinf(ll)):
        return -np.inf

    return lp + ll
