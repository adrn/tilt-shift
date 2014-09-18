# coding: utf-8

""" Given observations of Q = v sin i for some sample of objects, model the
    distribution of v as a mixture of components and infer parameters of those
    components. Essentially a deconvolution of sin i.
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys
import socket

# Third-party
import emcee
import matplotlib.pyplot as plt
import numpy as np
from astropy import log as logger
from streamteam.util import get_pool

def main(pool, path, plot=False):
    # Project
    from tiltshift.model import ln_posterior
    from tiltshift.fakedata import generate_data

    N = 256  # Number of fake data points
    J = 128  # Number of steps to use when integrating over i
    K = 16  # Number of mixture components

    base_save_path = os.path.join(path, "cache")
    save_path = os.path.join(base_save_path, "N{}_J{}_K{}".format(N,J,K))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # -------------------------------------------------------------------------
    # Generate fake data
    #

    np.random.seed(42)
    sigma = 1.  # km/s
    v_func = np.random.lognormal
    # v_func_args = dict(mean=5., sigma=0.01)  # tight v
    v_func_args = dict(mean=5., sigma=0.25)  # wide v
    v,Q,sigma_Q = generate_data(N, v_func, v_func_kwargs=v_func_args,
                                sigma=sigma, censor="Q > 5*sigma")

    # -------------------------------------------------------------------------
    # Now try sampling
    #

    # fix the positions of the mixture components
    v_k = np.linspace(5., 250., K)
    logger.debug("v_ks: {}".format(v_k))

    nwalkers = K*4
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
    pos,prob,state = sampler.run_mcmc(p0, 1000)

    sampler.reset()
    logger.debug("Running sampler for main sampling...")
    pos,prob,state = sampler.run_mcmc(pos, 5000)
    logger.debug("Done sampling!")

    pool.close()

    for j in range(ndim):
        plt.clf()
        for walker in sampler.chain[...,j]:
            plt.plot(walker)
        plt.ylim(0,sampler.flatchain.max())
        plt.xlabel("Step number")
        plt.savefig(os.path.join(save_path, "{}.png".format(j)))

    nbins = 25
    bins = np.linspace(v_k.min(), v_k.max(), nbins)

    fig,axes = plt.subplots(1, 2, figsize=(10,5), sharex=True)

    axes[0].hist(v, bins=bins)
    axes[0].set_xlabel(r'True $v$ [km s$^{-1}$]')

    median_a = np.median(sampler.flatchain, axis=0)
    axes[1].plot(v_k, median_a, linestyle='none', marker='o')
    axes[1].set_xlim(bins.min(), bins.max())
    axes[1].set_xlabel(r'Model')

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, "compare.png"))

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

    if socket.gethostname() == "adrn":
        logger.info("Assuming we're on adrn (laptop)...")
        base_path = "/Users/adrian/projects/tilt-shift"
    else:
        logger.info("Assuming we're on yeti...")
        base_path = "/vega/astro/users/amp2217/projects/tilt-shift"

    sys.path.append(base_path)
    main(pool, path=base_path, plot=args.plot)
    sys.exit(0)