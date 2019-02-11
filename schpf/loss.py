#!/usr/bin/env python

"""
Loss functions and higher order functions that return loss functions for a
given dataset

"""

import functools
import numpy as np

from schpf.hpf_numba import compute_pois_llh


#### Loss functions

def pois_llh_pointwise(X, *, theta, beta, **kwargs):
    """Poisson log-likelihood for each nonzero entry

    Parameters
    ----------
    X: coo_matrix
        Data to compute Poisson log likelihood of. Assumed to be nonzero.
    theta : HPF_Gamma
    beta : HPF_Gamma
    **kwargs : dict, optional
        extra arguments not used in this loss function

    Returns
    -------
    llh: ndarray


    Note
    ----
    Like all loss functions in this module, all parameters except from data
    must be passed to the function as a keyword argument, and the function
    will accept unused keyword args.
    """
    try:
        llh = compute_pois_llh(X.data, X.row, X.col,
                                theta.vi_shape, theta.vi_rate,
                                beta.vi_shape, beta.vi_rate)
    except NameError:
        e_rate = (theta.e_x[X.row] *  beta.e_x[X.col]).sum(axis=1)
        llh = X.data * np.log(e_rate) - e_rate - gammaln(X.data + 1)
    return llh


def mean_negative_pois_llh(X, *, theta, beta, **kwargs):
    """Mean Poisson log-likelihood for each nonzero entry

    Parameters
    ----------
    X: coo_matrix
        Data to compute Poisson log likelihood of. Assumed to be nonzero.
    theta : HPF_Gamma
    beta : HPF_Gamma
    **kwargs : dict, optional
        extra arguments not used in this loss function

    Returns
    -------
    llh: ndarray


    Note
    ----
    Like all loss functions in this module, all parameters except from data
    must be passed to the function as a keyword argument, and the function
    will accept unused keyword args.
    """
    return np.mean( -pois_llh_pointwise(X=X, theta=theta, beta=beta) )


# Higher-order functions
#
def get_loss_fnc_for_data(loss_function, X):
    return functools.partial(loss_function, X)


# # util
# #
# #

# def project_cells(Xcells, nfactors, a, ap, c, cp, dp, eta, beta):
    # cell_model = scHPF(nfactors=nfactors,
            # a=a, ap=ap, c=c, cp=cp, dp=dp, eta=eta, beta=beta)
    # return cell_model.project(vcells, replace=True)


# def get_loss_for_witheld_cells(loss_function, Xcells, nfactors, a, ap, c, cp, dp,
        # eta, beta):
    # """ Return a loss

    # """
    # def ret(a, ap, c, cp, dp, eta, beta):
        # project_cells

