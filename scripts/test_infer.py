# coding: utf-8

"""  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
# HACK:
sys.path.append("/Users/adrian/projects/tilt-shift/")

# Third-party
import matplotlib.pyplot as plt
import numpy as np
from astropy import log as logger

# Project
from tiltshift.model import ln_likelihood_explicit, ln_likelihood_fast
from tiltshift.fakedata import generate_data

# TODO: make plot of f(Q | {a_k}) for some choice of a_k's
def main(plot=False):
    np.random.seed(42)

    N = 100
    sigma = 5.  # km/s
    v_func = np.random.lognormal
    v_func_args = dict(mean=5., sigma=0.01)
    v,Q,sigma_Q = generate_data(N, v_func, v_func_kwargs=v_func_args,
                                sigma=sigma, censor="Q > 5*sigma")

    a_k = np.array([0.4, 0.5, 0.6])
    v_k = np.array([100, 150, 200.])

    ll1 = ln_likelihood_fast(a_k, v_k, Q, sigma_Q)
    ll1 = ll1.sum()

    ll2 = ln_likelihood_explicit(a_k, v_k, Q, sigma_Q)

    assert np.allclose(ll1, ll2, rtol=1E-1)

    QQs = np.linspace(50., 250., 100)
    sigma_QQs = np.ones_like(QQs)*sigma_Q[0]
    lls = ln_likelihood_fast(a_k, v_k, QQs, sigma_QQs)

    if plot:
        plt.clf()
        plt.plot(QQs, np.exp(lls-lls.max()))
        plt.show()

if __name__ == '__main__':
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")

    # parser.add_argument("-f", dest="field_id", default=None, required=True,
    #                     type=int, help="Field ID")
    parser.add_argument("-p", dest="plot", action="store_true", default=False,
                        help="Plot or not")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(plot=args.plot)
