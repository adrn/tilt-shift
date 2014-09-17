# coding: utf-8

"""  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import sys
# HACK:
sys.path.append("/Users/adrian/projects/tilt-shift/")

# Third-party
import emcee
import matplotlib.pyplot as plt
import numpy as np
from astropy import log as logger
from streamteam.util import get_pool

# Project
from tiltshift.model import ln_likelihood_explicit, ln_likelihood_fast, ln_posterior
from tiltshift.fakedata import generate_data

# TODO: make plot of f(Q | {a_k}) for some choice of a_k's
def main(pool, plot=False):
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

    # ------------------------------------------------------------------------
    # Now try actually sampling
    K = 10
    v_k = np.linspace(5., 200., K)
    logger.debug("v_ks: {}".format(v_k))

    nwalkers = 64
    ndim = K
    sampler = emcee.EnsembleSampler(nwalkers, dim=ndim,
                                    lnpostfn=ln_posterior,
                                    args=(v_k,Q,sigma_Q),
                                    pool=pool)
    p0 = np.random.normal(0.5,0.05,size=(nwalkers,ndim))
    p0 = p0 * N / p0.sum(axis=1)[:,np.newaxis]
    if np.any(p0 <= 0.):
        raise ValueError("Dumby.")

    logger.debug("Running sampler to burn in...")
    pos,prob,state = sampler.run_mcmc(p0, 100)

    sampler.reset()
    logger.debug("Running sampler for main sampling...")
    pos,prob,state = sampler.run_mcmc(pos, 100)
    logger.debug("Done sampling!")

    pool.close()
    sys.exit(0)

    for j in range(ndim):
        plt.clf()
        for walker in sampler.chain[...,j]:
            plt.plot(walker)
        plt.ylim(0,sampler.flatchain.max())
        plt.xlabel("Step number")
        plt.savefig("{}.png".format(j))

    nbins = 25
    bins = np.linspace(0,200,nbins)

    fig,axes = plt.subplots(1,2,figsize=(10,5),sharex=True)

    axes[0].hist(v, bins=bins)
    axes[0].set_xlabel(r'True $v$ [km s$^{-1}$]')

    median_a = np.median(sampler.flatchain, axis=0)
    vs = np.random.choice(v_k, p=median_a/np.sum(median_a), size=1000)
    axes[1].hist(vs, bins=bins)
    axes[1].set_xlim(bins.min(), bins.max())
    axes[1].set_xlabel(r'Model')
    fig.savefig("compare.png")

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
    parser.add_argument("--mpi", dest="mpi", default=False, action="store_true",
                        help="Run with MPI.")
    parser.add_argument("--threads", dest="threads", default=None, type=int,
                        help="Number of multiprocessing threads to run on.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    pool = get_pool(mpi=args.mpi, threads=args.threads)
    main(pool, plot=args.plot)
