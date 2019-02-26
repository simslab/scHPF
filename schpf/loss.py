#!/usr/bin/env python

"""
Loss functions and higher order functions that return loss functions for a
given dataset

"""

import functools
import numpy as np

from schpf.hpf_numba import compute_pois_llh

### Higher order loss functions

def loss_function_for_data(loss_function, X):
    """ Get a loss function for a fixed dataset

    Parameters
    ----------
    loss_function : function
        The loss function to use.  The data parameter for the function must
        be `X`
    X : coo_matrix
        coo_matrix of data to apply loss function to

    Returns
    -------
    fixed_data_loss_function : function
        A loss function which takes all the same parameters as the input
        `loss_function`, except for the data parameter `X` which is fixed
    """
    return functools.partial(loss_function, X=X)


def get_projection_loss_function(loss_function, X,
        model_kwargs={}, proj_kwargs={}):
    """ Project new data onto an existing model and calculate loss from it

    Parameters
    ----------
    loss_function : function
        the loss function to use on the projected data
    X : coo_matrix
        Data to project onto the existing model.  Can have an arbitrary number
        of rows (cells) > 0, but must have the same number of columns (genes)
        as the existing model
    model_kwargs : dict, optional
        additional keyword arguments for scHPF()
    proj_kwargs : dict, optional
        additional keyword arguments for scHPF.project(). By default,
        `max_iter`, 'min_iter', and 'check_freq'=5,


    Returns
    -------
    projection_loss_function : function
        A function which takes `a`, `ap`, `c`, `cp`, `dp`, `eta`, and `beta`
        for an scHPF model, projects a fixed dataset onto it, and takes the
        loss (using a fixed function) with respect to both the model and the
        data's projection.
    """
    # have to do import here to avoid issue with files importing each other
    from schpf import scHPF
    def projection_loss_function(*, a, ap, c, cp, dp, eta, beta, **kwargs):
        assert eta.vi_shape.shape[0] == beta.vi_shape[0]

        nfactors = beta.vi_shape[1]
        model = scHPF(nfactors=nfactors, a=a, ap=ap, c=c, cp=cp, dp=dp,
                    eta=eta, beta=beta, **model_kwargs)
        if 'max_iter' not in proj_kwargs: proj_kwargs['max_iter'] = 5
        if 'min_iter' not in proj_kwargs: proj_kwargs['min_iter'] = 5
        if 'cheq_freq' not in proj_kwargs: proj_kwargs['cheq_freq'] = 5
        model.project(X, replace=True, **proj_kwargs)

        return loss_function(X, a=model.a, ap=model.ap, bp=model.bp,
                c=model.c, cp=model.cp, dp=model.dp, xi=model.xi,
                eta=model.eta, theta=model.theta, beta=model.beta)

    return projection_loss_function


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
