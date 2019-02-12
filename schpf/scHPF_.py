#!/usr/bin/env python

import functools
from copy import deepcopy
from warnings import warn

import numpy as np
from scipy.sparse import coo_matrix
from scipy.misc import logsumexp
from scipy.special import digamma, gammaln, psi
from sklearn.base import BaseEstimator
from sklearn.externals import joblib

# TODO warn if can't import, and allow computation with slow
from schpf.hpf_numba import *


class HPF_Gamma(object):
    """Gamma variational distributions

    Parameters
    ----------
    vi_shape: np.ndarray
        Gamma shape parameter for the variational Gamma distributions.
        Ndarray.shape[0] must match `vi_rate`
    vi_rate: np.ndarray
        Gamma rate parameter for the variational Gamma distributions.
        Ndarray.shape[0] must match `vi_shape`
    """

    @staticmethod
    def random_gamma_factory(dims, shape_prior, rate_prior, dtype=np.float64):
        """Factory method to randomly initialize variational distributions

        Parameters
        ----------
        dims: list-like
            Numpy-style shape of the matrix of Gammas.
        shape_prior: float
            Prior for variational Gammas' shapes.  Must be greater than 0.
        rate_prior: float
            Prior for variational Gammas' rates.  Must be greater than 0.

        Returns
        -------
            A randomly initialized HPF_Gamma instance
        """
        vi_shape = np.random.uniform(0.5 * shape_prior, 1.5 * shape_prior,
                                     dims).astype(dtype)
        vi_rate  = np.random.uniform(0.5 * rate_prior, 1.5 * rate_prior,
                                     dims).astype(dtype)
        return HPF_Gamma(vi_shape,vi_rate)


    def __init__(self, vi_shape, vi_rate):
        """Initializes HPF_Gamma with variational shape and rates"""
        assert(vi_shape.shape == vi_rate.shape)
        assert(vi_shape.dtype == vi_rate.dtype)
        assert(np.all(vi_shape > 0))
        assert(np.all(vi_rate > 0))
        self.vi_shape = vi_shape
        self.vi_rate = vi_rate
        self.dims = vi_shape.shape
        self.dtype = vi_shape.dtype


    @property
    def e_x(self):
        """Expected value of the random variable(s) given variational
        distribution(s)
        """
        return self.vi_shape / self.vi_rate


    @property
    def e_logx(self):
        """Expectation of the log of random variable given variational
        distribution(s)"""
        return digamma(self.vi_shape) - np.log(self.vi_rate)


    @property
    def entropy(self):
        """Entropy of variational Gammas"""
        return  self.vi_shape - np.log(self.vi_rate) \
                + gammaln(self.vi_shape) \
                + (1 - self.vi_shape) * digamma(self.vi_shape)


    def sample(self, nsamples=1):
        """Sample from variational distributions

        Parameters
        ----------
        nsamples: int (optional, default 1)
            Number of samples to take.

        Returns
        -------
        X_rep : np.ndarray
            An ndarray of samples from the variational distributions, where
            the last dimension is the number of samples `nsamples`
        """
        samples = []
        for i in range(nsamples):
            samples.append(np.random.gamma(self.vi_shape, 1/self.vi_rate).T)
        return np.stack(samples).T


class scHPF(BaseEstimator):
    """HPF as described in ____

    Parameters
    ----------
    nfactors: int
        Number of factors (K)
    a: float, (optional, default 0.3)
        Hyperparameter a
    ap: float (optional, default 1.0)
        Hyperparameter a'
    bp: float (optional, default None)
        Hyperparameter b'. Set empirically from observed data if not
        given.
    c: float, (optional, default 0.3)
        Hyperparameter c
    cp: float (optional, default 1.0)
        Hyperparameter c'
    dp: float (optional, default None)
        Hyperparameter d'. Set empirically from observed data if not
        given.
    min_iter: int (optional, default 30):
        Minimum number of interations for training.
    max_iter: int (optional, default 1000):
        Maximum number of interations for training.
    check_freq: int (optional, default 10)
        Number of training iterations between calculating loss.
    epsilon: float (optional, default 0.001)
        Percent change of loss for convergence.
    better_than_n_ago: int (optional, default 5)
        Stop condition if loss is getting worse.  Stops training if loss
        is worse than `better_than_n_ago`*`check_freq` training steps
        ago and getting worse.
    xi: HPF_Gamma (optional, default None)
        Variational distributions for xi
    theta: HPF_Gamma (optional, default None)
        Variational distributions for theta
    eta: HPF_Gamma (optional, default None)
        Variational distributions for eta
    beta: HPF_Gamma (optional, default None)
        Variational distributions for beta
    verbose: bool (optional, default True)
            Print messages at each check_freq
    """
    def __init__(
            self,
            nfactors,
            a=0.3,
            ap=1,
            bp=None,
            c=0.3,
            cp=1,
            dp=None,
            min_iter=30,
            max_iter=1000,
            check_freq=10,
            epsilon=0.001,
            better_than_n_ago=5,
            dtype=np.float64,
            xi=None,
            theta=None,
            eta=None,
            beta=None,
            loss=None,
            verbose=True,
            ):
        """Initialize HPF instance"""
        self.nfactors = nfactors
        self.a = a
        self.ap = ap
        self.bp = bp
        self.c = c
        self.cp = cp
        self.dp = dp
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.check_freq = check_freq
        self.epsilon = epsilon
        self.better_than_n_ago = better_than_n_ago
        self.dtype = dtype
        self.verbose = verbose

        self.xi = None
        self.eta = None
        self.theta = None
        self.beta = None

        self.loss = []


    @property
    def ngenes(self):
        return self.eta.dims[0] if self.eta is not None else None


    @property
    def ncells(self):
        return self.xi.dims[0] if self.xi is not None else None


    def cell_score(self, xi=None, theta=None):
        """Get cell score from xi and theta

        Properties
        ----------
        xi : HPF_Gamma, (optional, default self.xi)
            varitional distributions for xi
        theta : HPF_Gamma, (optional, default self.theta)
            varitional distributions for theta

        Returns
        -------
        cell_score : ndarray
            ncell x nfactor array of cell scores
        """
        xi = self.xi if xi is None else xi
        theta = self.theta if theta is None else theta
        return self._score(xi, theta)


    def gene_score(self, eta=None, beta=None):
        """Get cell score from eta and beta

        Parameters
        ----------
        eta : HPF_Gamma, (optional, default self.eta)
            varitional distributions for eta
        beta : HPF_Gamma, (optional, default self.beta)
            varitional distributions for beta

        Returns
        -------
        gene_score : ndarray
            ngene x nfactor array of cell scores
        """
        eta = self.eta if eta is None else eta
        beta = self.beta if beta is None else beta
        return self._score(eta, beta)


    def pois_llh_pointwise(self, X, theta=None, beta=None):
        """Poisson log-likelihood (for each nonzero data)

        Attempt to use numba/cffi/gsl, use numpy otherwise

        Parameters
        ----------
        X: coo_matrix
            Data to compute Poisson log likelihood of. Assumed to be nonzero.
        theta : HPF_Gamma, optional
            If given, use for theta instead of self.theta
        beta : HPF_Gamma, optional
            If given, use for beta instead of self.beta

        Returns
        -------
        llh: ndarray
        """
        theta = self.theta if theta is None else theta
        beta = self.beta if beta is None else beta
        return pois_llh_pointwise(X=X, theta=theta, beta=beta)


    def mean_negative_pois_llh(X, theta=None, beta=None, **kwargs):
        """Convenience method for mean negative llh of nonzero entries

        """
        theta = self.theta if theta is None else theta
        beta = self.beta if beta is None else beta
        return mean_negative_pois_llh(X=X, theta=theta, beta=beta)


    def fit(self, X, **params):
        """Fit an scHPF model

        Parameters
        ----------
        X: coo_matrix
            Data to fit
        validation_nz: coo_matrix, (optional, default None)
            coo_matrix, which should have the same shape as
        """
        (bp, dp, xi, eta, theta, beta, loss) = self._fit(
                X, **params)
        self.bp = bp
        self.dp = dp
        self.xi = xi
        self.eta = eta
        self.theta = theta
        self.beta = beta
        self.loss = loss
        return self


    def project(self, X, replace=False, min_iter=4, max_iter=14, check_freq=2,
            **kwargs):
        """Project new cells into latent space

        Gene distributions (beta and eta) are fixed.

        Parameters
        ----------
        X: coo_matrix
            Data to project.  Should have self.ngenes columns
        replace: bool (optional, default False)
            Replace theta, xi, and bp with projected values in self
        min_iter: int (optional, default 4)
            Replaces self.min_iter if not None. Few iterations are needed
            because beta and eta are fixed.
        max_iter: int (optional, default 14)
            Replaces self.max_iter if not None. Few iterations are needed
            because beta and eta are fixed.
        check_freq: int (optional, default 2)
            Number of training iterations between calculating loss.

        Returns
        -------
        projection : scHPF
            An scHPF object with variational distributions theta and xi (for the
            new cells in `X`) and the same variational distributions as self
            for gene distributions beta and eta

        """
        (bp, _, xi, _, theta, _) = self._fit(X,
                min_iter=min_iter, max_iter=max_iter, check_freq=check_freq,
                freeze_genes=True)
        if replace:
            self.bp = bp
            self.xi = xi
            self.theta = theta
            return self
        else:
            new_scHPF = deepcopy(self)
            new_scHPF.bp = bp
            new_scHPF.xi = xi
            new_scHPF.theta = theta
            return new_scHPF


    def _score(self, capacity, loading):
        """Get hierarchically normalized gene or cell loadings"""
        return loading.e_x * capacity.e_x[:,None]


    def _fit(self, X, freeze_genes=False, reinit=True, min_iter=None,
            loss_function=None, checkstep_function=None, verbose=None):
        """Combined internal fit/transform function

        Parameters
        ----------
        X: coo_matrix
            Data to fit
        freeze_genes: bool, (optional, default False)
            Should we update gene variational distributions eta and beta
        reinit: bool, (optional, default True)
            Randomly initialize variational distributions even if they
            already exist. Superseded by freeze_genes.
        min_iter: int (optional, default None)
            Replaces self.min_iter if given.  Useful when projecting
            new data onto an existing scHPF model.
        loss_function : function, (optional, default None)
            Function to use for loss, which is assumed to be nonzero and
            decrease with improvement. Must accept hyperparameters a, ap,
            bp, c, cp, and dp and the variational distributions for xi, eta,
            theta, and beta even if only some of these values are used. Should
            have an internal reference to any data used (_fit will not pass it
            any data). If `loss_function` is not given/None, the mean negative
            log likelihood of nonzero values in training data `X` is used.
        checkstep_function : function  (optional, default None)
            A function that takes arguments bp, dp, xi, eta, theta, beta, and
            t and, if given, is called at check_interval. Intended use is
            to check additional stats during training, potentially with
            hardcoded data, but is unrestricted.  Use at own risk.
        verbose: bool (optional, default None)
            If not None, overrides self.verbose

        Returns
        -------
        bp: float
            Empirically set value for bp
        dp: float
            Empirically set value for dp. Unchanged if freeze_genes.
        xi: HPF_Gamma
            Learned variational distributions for xi
        eta: HPF_Gamma
            Learned variational distributions for eta. Unchanged if
            freeze_genes.
        theta: HPF_Gamma
            Learned variational distributions for theta
        beta: HPF_Gamma
            Learned variational distributions for beta. Unchanged if
            freeze_genes.
        loss : list
            loss at each checkstep
        """
        # local (convenience) vars for model
        nfactors, (ncells, ngenes) = self.nfactors, X.shape
        a, ap, c, cp = self.a, self.ap, self.c, self.cp

        # get empirically set hyperparameters and variational distributions
        bp, dp, xi, eta, theta, beta = self._setup(X, freeze_genes, reinit)

        # Make first updates for hierarchical shape prior
        # (vi_shape is constant, but want to update full distribution)
        xi.vi_shape[:] = ap + nfactors * a
        if not freeze_genes:
            eta.vi_shape[:] = cp + nfactors * c

        # setup loss function as mean negative llh of nonzero training data
        # if the loss function is not given
        if loss_function is None:
            loss_function = loss_function_for_data(
                    mean_negative_pois_llh, X)

        ## init
        loss, pct_change = [], []
        min_iter = self.min_iter if min_iter is None else min_iter
        verbose = self.verbose if verbose is None else verbose
        for t in range(self.max_iter):
            if t==0 and reinit: #randomize phi for first iteration
                random_phi = np.random.dirichlet( np.ones(nfactors),
                        X.data.shape[0])
                Xphi_data = X.data[:,None] * random_phi
            else:
                Xphi_data = compute_Xphi_data(X.data, X.row, X.col,
                                            theta.vi_shape, theta.vi_rate,
                                            beta.vi_shape, beta.vi_rate)

            # gene updates (if not frozen)
            if not freeze_genes:
                beta.vi_shape = compute_loading_shape_update(Xphi_data, X.col,
                        ngenes, c)
                beta.vi_rate = compute_loading_rate_update(eta.vi_shape,
                        eta.vi_rate, theta.vi_shape, theta.vi_rate)
                eta.vi_rate = dp + beta.e_x.sum(1)

            # cell updates
            theta.vi_shape = compute_loading_shape_update(Xphi_data, X.row,
                                                          ncells, a)
            theta.vi_rate = compute_loading_rate_update(xi.vi_shape, xi.vi_rate,
                    beta.vi_shape, beta.vi_rate)
            xi.vi_rate = bp + theta.e_x.sum(1)


            # record llh/percent change and check for convergence
            if t % self.check_freq == 0:

                # chech llh
                # vX = validation_data if validation_data is not None else X
                try :
                    curr = loss_function(
                                a=a, ap=ap, bp=bp, c=c, cp=cp, dp=dp,
                                xi=xi, eta=eta, theta=theta, beta=beta)
                    loss.append(curr)
                except NameError as e:
                    print('Invalid loss function')
                    raise e

                # calculate percent change
                try:
                    prev = loss[-2]
                    pct_change.append(100 * (curr - prev) / np.abs(prev))
                except IndexError:
                    pct_change.append(100)
                if verbose:
                    msg = '[Iter. {0: >4}]  loss:{1:.6f}  pct:{2:.9f}'.format(
                            t, curr, pct_change[-1])
                    print(msg)
                if checkstep_function is not None:
                    checkstep_function(bp, dp, xi, eta, theta, beta, t)

                # check convergence
                if len(loss) > 3 and t >= min_iter:
                    # convergence conditions (all must be met)
                    current_small = np.abs(pct_change[-1]) < self.epsilon
                    prev_small = np.abs(pct_change[-2]) < self.epsilon
                    not_inflection = not (
                            (np.abs(loss[-3]) < np.abs(prev)) \
                            and (np.abs(prev) > np.abs(curr)))
                    converged = current_small and prev_small and not_inflection
                    if converged:
                        if verbose:
                            print('converged')
                        break

                    # getting worse, and has been for better_than_n_ago checks
                    # (don't waste time on a bad run)
                    if len(loss) > self.better_than_n_ago \
                            and self.better_than_n_ago:
                        nprev = loss[-self.better_than_n_ago]
                        worse_than_n_ago = np.abs(nprev) < np.abs(curr)
                        getting_worse = np.abs(prev) < np.abs(curr)
                        if worse_than_n_ago and getting_worse:
                            if verbose:
                                print('getting worse break')
                            break


            # TODO message or warning or something
            if t >= self.max_iter:
                break

        return (bp, dp, xi, eta, theta, beta, loss)


    def _setup(self, X, freeze_genes=False, reinit=True, clip=True):
        """Setup variational distributions"""
        # locals for convenience
        nfactors, (ncells, ngenes) = self.nfactors, X.shape
        a, ap, c, cp = self.a, self.ap, self.c, self.cp
        bp, dp = self.bp, self.dp

        xi, eta, theta, beta = (self.xi, self.eta, self.theta, self.beta)

        # empirically set bp and dp
        def mean_var_ratio(X, axis):
            axis_sum = X.sum(axis=axis)
            return np.mean(axis_sum) / np.var(axis_sum)
        if bp is None:
            bp = ap * mean_var_ratio(X, axis=1)
        if dp is None: # dp first in case of error
            if freeze_genes:
                msg = 'dp is None and cannot  dp when freeze_genes is True.'
                raise ValueError(msg)
            else:
                dp = cp *  mean_var_ratio(X, axis=0)
                if clip and bp > 1000 * dp:
                    old_val = dp
                    dp = bp / 1000
                    print('Clipping dp: was {} now {}'.format(old_val, dp))

        if reinit or (xi is None):
            xi = HPF_Gamma.random_gamma_factory((ncells,), ap, bp,
                    dtype=self.dtype)
        if reinit or (theta is None):
            theta = HPF_Gamma.random_gamma_factory((ncells,nfactors), a, bp,
                    dtype=self.dtype)

        # Check if variational distributions for genes exist, create if not
        # Error if freeze_genes and eta and beta don't exists
        if freeze_genes:
            if eta is None or beta is None:
                msg = 'To fit with frozen gene variational distributions ' \
                    + '(`freeze_genes`==True), eta and beta must be set to ' \
                    + 'valid HPF_Gamma instances.'
                raise ValueError(msg)
        else:
            if reinit or (eta is None):
                eta = HPF_Gamma.random_gamma_factory((ngenes,), cp, dp,
                        dtype=self.dtype)
            if reinit or (beta is None):
                beta = HPF_Gamma.random_gamma_factory((ngenes,nfactors),
                        c, dp, dtype=self.dtype)

        return (bp, dp, xi, eta, theta, beta)


    def _initialize(self, X, freeze_genes=False):
        """Shortcut to setup random distributions without fitting"""
        bp, dp, xi, eta, theta, beta = self._setup(X, freeze_genes,
                reinit=True)
        self.bp = bp
        self.dp = dp
        self.xi = xi
        self.eta = eta
        self.theta = theta
        self.beta = beta


def load_model(file_name):
    """Load a model from a joblib file

    Parameters
    ----------
    file_name : str
        Joblib file containing a saved scHPF model
    """
    return joblib.load(file_name)


def save_model(model, file_name):
    """Save model to (joblib) file

    Serialize scHPF model as a joblib file.  Joblib is simillar to pickle,
    but preferable for objects with many numpy arrays

    Parameters
    ----------
    model : scHPF
        The scHPF model object to save
    file_name : str
        Name of file to save model to
    """
    joblib.dump(model, file_name)


def run_trials(X, nfactors,
        ntrials=5,
        min_iter=30,
        max_iter=1000,
        check_freq=10,
        epsilon=0.001,
        better_than_n_ago=5,
        dtype=np.float64,
        verbose=True,
        ):
    """
    Train with multiple random initializations, selecting model with best loss

    As scHPF uses non-convex optimization, it benefits from training with
    multiple random initializations to avoid local minima.

    Parameters
    ----------
    X: coo_matrix
        Data to fit
    nfactors: int
        Number of factors (K)
    ntrials : int (optional, default 5)
        Number of random initializations for training
    min_iter: int (optional, default 30):
        Minimum number of interations for training.
    max_iter: int (optional, default 1000):
        Maximum number of interations for training.
    check_freq: int (optional, default 10)
        Number of training iterations between calculating loss.
    epsilon: float (optional, default 0.001)
        Percent change of loss for convergence.
    better_than_n_ago: int (optional, default 5)
        Stop condition if loss is getting worse.  Stops training if loss
        is worse than `better_than_n_ago`*`check_freq` training steps
        ago and getting worse.
    """
    ngenes = X.shape[1]
    if ngenes >= 20000:
        msg = 'WARNING: you are running scHPF with {} genes,'.format(ngenes)
        msg += ' which is more than the ~20k protein coding genes in the'
        msg += ' human genome. We suggest running scHPF on protein-coding'
        msg += ' genes only.'
        print(msg)

    best_loss, best_model, best_t = np.finfo(np.float64).max, None, None
    for t in range(ntrials):
        model = scHPF(nfactors=nfactors,
                    min_iter=min_iter, max_iter=max_iter,
                    check_freq=check_freq, epsilon=epsilon,
                    better_than_n_ago=better_than_n_ago,
                    verbose=verbose, dtype=dtype,
                    )
        model.fit(X)

        loss = model.loss[-1]
        if loss < best_loss:
            best_model = model
            best_loss = loss
            best_t = t
            if verbose:
                print('New best!'.format(t))
        if verbose:
            print('Trial {0} loss: {1:.6f}'.format(t, loss))
            print('Best loss: {0:.6f} (trial {1})'.format(best_loss, best_t))

    return best_model



"""
Loss functions and higher order functions that return loss functions for a
given dataset

Annoyance
---------
I would love to put everything below in a separate file but there's some weird
import recursion that I can't figure out unless I pass
get_projection_loss_function an scHPF object rather than creating it within
the method, (which I do not want to do).
"""

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


def get_projection_loss_function(loss_function, X, *,
        a, ap, c, cp, dp, eta, beta,
        model_kwargs={}, proj_kwargs={},
        **kwargs):
    """ Project new data onto an existing model and calculate loss from it

    Parameters
    ----------
    loss_function : function
        the loss function to use on the projected data
    X : coo_matrix
        Data to project onto the existing model.  Can have an arbitrary number
        of rows (cells) > 0, but must have the same number of columns (genes)
        as the existing model
    a : int
        hyperparameter `a` in the existing model
    ap : int
        hyperparameter `ap` in the existing model
    c : int
        hyperparameter `c` in the existing model
    cp : int
        hyperparameter `cp` in the existing model
    dp : int
        hyperparameter `dp` in the existing model
    eta : HPF_Gamma
        variational distribution for eta in the existing model
    beta : HPF_Gamma
        variational distribution for beta in the existing model

    Returns
    -------
    projection_loss_function : function
        A function which takes `a`, `ap`, `c`, `cp`, `dp`, `eta`, and `beta`
        for an scHPF model, projects a fixed dataset onto it, and takes the
        loss (using a fixed function) with respect to both the model and the
        data's projection.
    """
    def projection_loss_function(*, a, ap, c, cp, dp, eta, beta, **kwargs):
        assert eta.vi_shape.shape[0] == beta.vi_shape[0]

        nfactors = beta.vi_shape[1]
        model = scHPF(nfactors=nfactors, a=a, ap=ap, c=c, cp=cp, dp=dp,
                    eta=eta, beta=beta, **model_kwargs)
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
