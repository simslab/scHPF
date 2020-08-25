#!/usr/bin/env python
from copy import deepcopy
from warnings import warn
from functools import partial
from multiprocessing import cpu_count

import numpy as np
from scipy.sparse import coo_matrix
from scipy.special import digamma, gammaln, psi
try:
    from scipy.misc import logsumexp
except ImportError:
    from scipy.special import logsumexp

from sklearn.base import BaseEstimator
import joblib
from joblib import Parallel, delayed


# TODO warn if can't import, and allow computation with slow
from schpf.hpf_numba import *
from schpf.util import minibatch_ix_generator
import schpf.loss as ls
import schpf


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

    Attributes
    ----------
    vi_shape : ndarray
    vi_rate : ndarray
    dims : ndarray
        The shape of vi_shape and vi_rate
    dtype : dtype
        dtype of both vi_shape and vi_rate
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
        self.dtype = vi_shape.dtype


    def __eq__(self, other):
        if isinstance(other, self.__class__):
            shape_equal = np.array_equal(self.vi_shape, other.vi_shape)
            rate_equal = np.array_equal(self.vi_rate, other.vi_rate)
            dtype_equal = self.dtype == other.dtype
            return shape_equal and rate_equal and dtype_equal
        return False


    @property
    def dims(self):
        assert self.vi_shape.shape == self.vi_rate.shape
        return self.vi_shape.shape


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


    def combine(self, other, other_ixs):
        """ Combine with another HPF_Gamma

        Useful for combining variational distributions from training data with
        variational distributions from cells that were projected onto frozen
        beta and eta

        Parameters
        ----------
        other : `HPF_Gamma`
            Other HPF_Gamma to merge with
        other_ixs : list or ndarray
            Ordered indices of other in the merged HPF_Gamma. Must have len
            equal to other.shape[0]. Must have a maximum value less than
            self.dims[0] + other.shape[0]. May not have any repeat values.

        Returns
        -------
        combined_model : `HPF_Gamma`
        """
        assert other.dims[0] == len(other_ixs)
        assert len(np.unique(other_ixs)) == len(other_ixs)
        assert self.dims[0] + other.dims[0] > np.max(other_ixs)

        new_dims = [self.dims[0]+other.dims[0], *self.dims[1:]]
        self_ixs = np.setdiff1d(np.arange(new_dims[0]),
                other_ixs)

        new_vi_shape = np.empty(new_dims, dtype=self.dtype)
        new_vi_shape[self_ixs] = self.vi_shape
        new_vi_shape[other_ixs] = other.vi_shape

        new_vi_rate = np.empty(new_dims, dtype=self.dtype)
        new_vi_rate[self_ixs] = self.vi_rate
        new_vi_rate[other_ixs] = other.vi_rate

        return HPF_Gamma(new_vi_shape, new_vi_rate)


class scHPF(BaseEstimator):
    """scHPF as described in Levitin et al., Molecular Systems Biology 2019

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
            loss=[],
            verbose=True,
            ):
        """Initialize HPF instance"""
        self.version = schpf.__version__
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

        self.xi = xi
        self.eta = eta
        self.theta = theta
        self.beta = beta

        self.loss = []


    @property
    def ngenes(self):
        return self.eta.dims[0] if self.eta is not None else None


    @property
    def ncells(self):
        return self.xi.dims[0] if self.xi is not None else None


    def cell_score(self, xi=None, theta=None):
        """Get cell score from xi and theta

        Parameters
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
        return ls.pois_llh_pointwise(X=X, theta=theta, beta=beta)


    def mean_negative_pois_llh(self, X, theta=None, beta=None, **kwargs):
        """Convenience method for mean negative llh of nonzero entries

        """
        theta = self.theta if theta is None else theta
        beta = self.beta if beta is None else beta
        return ls.mean_negative_pois_llh(X=X, theta=theta, beta=beta)


    def fit(self, X, **kwargs):
        """Fit an scHPF model

        Parameters
        ----------
        X: coo_matrix
            Data to fit
        loss_function : function, optional (Default: None)
            loss function to use for fit. set to negative poisson likelihood
            of X if not given
        """
        (bp, dp, xi, eta, theta, beta, loss) = self._fit(
                X, **kwargs)
        self.bp = bp
        self.dp = dp
        self.xi = xi
        self.eta = eta
        self.theta = theta
        self.beta = beta
        self.loss = loss
        return self


    def project(self, X, recalc_bp=False, replace=False, min_iter=2, max_iter=50,
            check_freq=2, **kwargs):
        """Project new cells into latent space

        Gene distributions (beta and eta) are fixed.

        Parameters
        ----------
        X: coo_matrix
            Data to project.  Should have self.ngenes columns
        recalc_bp : bool, optional (Default: False)
            Recalculated value of empirical hyperparameter bp. Do not do this
            for withheld text data.
        replace: bool, optional (Default: False)
            Replace theta and xi with projected values in self. Note that
            loss will not be updated
        min_iter: int, (Default: 2)
            Replaces self.min_iter if not None. Few iterations are needed
            because beta and eta are fixed.
        max_iter: int, (Default: 10)
            Replaces self.max_iter if not None. Few iterations are needed
            because beta and eta are fixed.
        check_freq: int, optional (Default: 2)
            Number of training iterations between calculating loss.

        Returns
        -------
        result : scHPF or ndarray
            If replace=`False`, an  scHPF object with variational
            distributions theta and xi (for the new cells in `X`) and the
            same variational distributions as self for gene distributions
            beta and eta. If replace=`True`, then the loss for the projection
            (xi and theta will be updated in self but not returned). In both
            cases, bp will only be updated for the new data if self.bp==None or
            recalc_bp=`True`.

        """
        if replace and recalc_bp:
            msg = 'Cannot replace `bp` with recalculated value'
            raise ValueError(msg)

        model = self if replace else deepcopy(self)
        if recalc_bp: model.bp = None
        (bp, _, xi, _, theta, _, loss) = model._fit(X,
                min_iter=min_iter, max_iter=max_iter, check_freq=check_freq,
                freeze_genes=True, **kwargs)
        if replace:
            self.xi = xi
            self.theta = theta
            return loss
        else:
            model.bp = bp
            model.xi = xi
            model.theta = theta
            model.loss = loss
            return model


    def _score(self, capacity, loading):
        """Get the hierarchically normalized loadings which we call the cell
        or gene score in the scHPF paper

        Parameters
        ----------
        capacity : HPF_Gamma
            xi or eta
        loading : HPF_Gamma
            theta or beta


        Returns
        -------
        score : ndarray
        """
        assert(loading.dims[0] == capacity.dims[0])
        return loading.e_x * capacity.e_x[:,None]


    def _fit(self, X, freeze_genes=False, reinit=True, loss_function=None,
            min_iter=None, max_iter=None, epsilon=None, check_freq=None,
            single_process=False, checkstep_function=None, verbose=None,
            batchsize=None, beta_theta_simultaneous=False,
            loss_smoothing=1):
        """Combined internal fit/transform function

        Parameters
        ----------
        X: coo_matrix
            Data to fit
        freeze_genes: bool, (optional, default False)
            Should we update gene variational distributions eta and beta
        reinit: bool, (optional, default True)
            Randomly initialize variational distributions even if they
            already exist. Superseded by freeze_genes. Does not affect
            self.bp and self.dp which will only be set empirically if they
            are None
        loss_function : function, (optional, default None)
            Function to use for loss, which is assumed to be nonzero and
            decrease with improvement. Must accept hyperparameters a, ap,
            bp, c, cp, and dp and the variational distributions for xi, eta,
            theta, and beta even if only some of these values are used.
            Should have an internal reference to any data used (_fit will
            not pass it any data). If `loss_function` is not given or equal
            to None, the mean negative log likelihood of nonzero values in
            training data `X` is used.
        min_iter: int (optional, default None)
            Replaces self.min_iter if given.  Useful when projecting
            new data onto an existing scHPF model.
        max_iter: int (optional, default None)
            Replaces self.max_iter if given.  Useful when projecting
            new data onto an existing scHPF model.
        epsilon: float (optional, default None)
            Replaces self.epsilon if given. Percent change of loss for
            convergence.
        check_freq : int, optional (Default: None)
            Replaces self.check_freq if given.  Useful when projecting
            new data onto an existing scHPF model.
        single_process : bool, optional (Default: False)
            Use single-threaded versions of updates
        checkstep_function : function  (optional, default None)
            A function that takes arguments bp, dp, xi, eta, theta, beta,
            and t and, if given, is called at check_interval. Intended use
            is to check additional stats during training, potentially with
            hardcoded data, but is unrestricted.  Use at own risk.
        verbose: bool (optional, default None)
            If not None, overrides self.verbose
        batchsize: int, optional (Default 0)
            number of cells per batch. When 0, all cells are used
        beta_theta_simultaneous: bool, option (Default False)
            Should updates for beta and theta be computed simultaneously.
            If False, beta is updated first, and theta is updated using
            that beta
        loss_smoothing: int, optional (Default: 1)
            Smooth loss up to `loss_smoothing` check frequencies ago. 1 results
            in no smoothing. Intended to be used with batching when assessing
            convergence based on training loss, where a good value might be
            int(ncells/n_batches)

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
        assert loss_smoothing > 0

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
            loss_function = ls.loss_function_for_data(
                    ls.mean_negative_pois_llh, X)

        # setup batch_ix iterator
        if batchsize is not None and batchsize > 1 and batchsize <= ncells:
            batched = True
            batch_ix_generator = minibatch_ix_generator(ncells, batchsize)
        else:
            batched = False
            batch_ix_generator = None

        ## init
        loss, unsmoothed_loss, pct_change = [], [], []
        # check variable overrides
        min_iter = self.min_iter if min_iter is None else min_iter
        max_iter = self.max_iter if max_iter is None else max_iter
        epsilon = self.epsilon if epsilon is None else epsilon
        check_freq = self.check_freq if check_freq is None else check_freq
        verbose = self.verbose if verbose is None else verbose
        for t in range(max_iter):
            # setup batching
            if batch_ix_generator is None:
                batch_ix = np.arange(X.shape[0])
                batchsize = ncells
                X_batch = X
            else:
                batch_ix = next(batch_ix_generator)
                X_batch = X.tocsr()[batch_ix,:].tocoo()

            if t==0 and reinit: #randomize phi for first iteration
                random_phi = np.random.dirichlet( np.ones(nfactors),
                        X_batch.data.shape[0])
                Xphi_data = X_batch.data[:,None] * random_phi
            else:
                if single_process:
                    Xphi_data = compute_Xphi_data_numpy(X_batch, theta, beta,
                            theta_ix=batch_ix)
                else:
                    Xphi_data = compute_Xphi_data(
                            X_batch.data, X_batch.row, X_batch.col,
                            theta.vi_shape[batch_ix], theta.vi_rate[batch_ix],
                            beta.vi_shape, beta.vi_rate)

            if beta_theta_simultaneous:
                # calculate gene updates but don't assign yet
                if not freeze_genes:
                    bvs = compute_loading_shape_update(Xphi_data,
                            X_batch.col, ngenes, c)
                    bvr = compute_loading_rate_update(eta.vi_shape,
                            eta.vi_rate, theta.vi_shape[batch_ix],
                            theta.vi_rate[batch_ix])
                # cell updates
                theta.vi_shape[batch_ix] = compute_loading_shape_update(
                        Xphi_data, X_batch.row, batchsize, a)
                theta.vi_rate[batch_ix] = compute_loading_rate_update(
                        xi.vi_shape[batch_ix], xi.vi_rate[batch_ix],
                        beta.vi_shape, beta.vi_rate)
                xi.vi_rate[batch_ix] = bp + theta.e_x[batch_ix].sum(1)
                # make gene updates
                if not freeze_genes:
                    beta.vi_shape = bvs
                    beta.vi_rate = bvr
                    eta.vi_rate = dp + beta.e_x.sum(1)

            else:
                if batched:
                    # cell updates, must do first for batching
                    theta.vi_shape[batch_ix] = compute_loading_shape_update(
                            Xphi_data, X_batch.row, batchsize, a)
                    theta.vi_rate[batch_ix] = compute_loading_rate_update(
                            xi.vi_shape[batch_ix], xi.vi_rate[batch_ix],
                            beta.vi_shape, beta.vi_rate)
                    xi.vi_rate[batch_ix] = bp + theta.e_x[batch_ix].sum(1)

                if not freeze_genes:
                    #gene updates
                    beta.vi_shape = compute_loading_shape_update(Xphi_data,
                            X_batch.col, ngenes, c)
                    beta.vi_rate = compute_loading_rate_update(eta.vi_shape,
                            eta.vi_rate, theta.vi_shape[batch_ix],
                            theta.vi_rate[batch_ix])
                    eta.vi_rate = dp + beta.e_x.sum(1)

                if not batched:
                    # cell updates, doing after gene updates when not batched
                    # for legacy consistency
                    theta.vi_shape[batch_ix] = compute_loading_shape_update(
                            Xphi_data, X_batch.row, batchsize, a)
                    theta.vi_rate[batch_ix] = compute_loading_rate_update(
                            xi.vi_shape[batch_ix], xi.vi_rate[batch_ix],
                            beta.vi_shape, beta.vi_rate)
                    xi.vi_rate[batch_ix] = bp + theta.e_x[batch_ix].sum(1)


            # record llh/percent change and check for convergence
            if t % check_freq == 0:

                # chech llh
                # vX = validation_data if validation_data is not None else X
                try :
                    curr = loss_function(
                                a=a, ap=ap, bp=bp, c=c, cp=cp, dp=dp,
                                xi=xi, eta=eta, theta=theta, beta=beta)
                    unsmoothed_loss.append(curr)
                    if len(unsmoothed_loss) > loss_smoothing:
                        unsmoothed_loss = unsmoothed_loss[1:]
                    # normally this is just curr as loss_smoothing=1 by default
                    loss.append(np.mean(unsmoothed_loss))
                except NameError as e:
                    print('Invalid loss function')
                    raise e

                # calculate percent change
                try:
                    curr, prev = loss[-1], loss[-2]
                    pct_change.append(100 * (curr - prev) / np.abs(prev))
                except IndexError:
                    pct_change.append(100)
                if verbose:
                    msg = '[Iter. {0: >4}]  loss:{1:.6f}  pct:{2:.9f}'.format(
                            t, curr, pct_change[-1])
                    print(msg)
                if checkstep_function is not None:
                    checkstep_function(bp=bp, dp=dp, xi=xi, eta=eta, theta=theta,
                            beta=beta, t=t)

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
                        nprev = loss[-self.better_than_n_ago] \
                                if len(loss)>self.better_than_n_ago else loss[0]
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
        """Setup variational distributions

        Parameters
        ----------
        X: coo_matrix
            Data to fit
        freeze_genes: bool, optional (Default: False)
            Should we update gene variational distributions eta and beta
        reinit: bool, optional (Default: True)
            Randomly initialize variational distributions even if they
            already exist. Superseded by freeze_genes. Does not affect
            self.bp and self.dp (which will only be set empirically if
            they are None)
        clip : bool, optional (Default: True)
            If empirically calculating dp and bp > 1000 * dp, clip dp to
            bp / 1000.

        Returns
        -------
        bp : float
        dp : float
        xi : HPF_Gamma
        eta : HPF_Gamma
        theta : HPF_Gamma
        beta : HPF_Gamma

        """
        # locals for convenience
        nfactors, (ncells, ngenes) = self.nfactors, X.shape
        a, ap, c, cp = self.a, self.ap, self.c, self.cp
        bp, dp = self.bp, self.dp

        xi, eta, theta, beta = (self.xi, self.eta, self.theta, self.beta)

        # empirically set bp and dp
        bp, dp = self._get_empirical_hypers(X, freeze_genes, clip)

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


    def _get_empirical_hypers(self, X, freeze_genes=False, clip=True):
        """Get empirical values for bp, dp

        Parameters
        ----------
        X : coo_matrix
            Data to fit

        Returns
        -------
        bp : float
        dp : float
        """
        bp, dp = self.bp, self.dp
        # empirically set bp and dp
        def mean_var_ratio(X, axis):
            axis_sum = X.sum(axis=axis)
            return np.mean(axis_sum) / np.var(axis_sum)
        if bp is None:
            bp = self.ap * mean_var_ratio(X, axis=1)
        if dp is None: # dp first in case of error
            if freeze_genes:
                msg = 'dp is None and cannot be set'
                msg += ' when freeze_genes is True.'
                raise ValueError(msg)
            else:
                dp = self.cp *  mean_var_ratio(X, axis=0)
                if clip and bp > 1000 * dp:
                    old_val = dp
                    dp = bp / 1000
                    print('Clipping dp: was {} now {}'.format(old_val, dp))

        return bp, dp


    def _initialize(self, X, freeze_genes=False):
        """Shortcut to setup random distributions & set variables
        """
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


    Returns
    -------
    model : scHPF
        The scHPF model in the file
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


def combine_across_cells(x, y, y_ixs):
    """Combine theta & xi from two scHPF instance with the same beta & eta

    Intended to be used combining variational distributions for local
    variables (theta,xi) from training data with variational distributions
    for local variables from validation or other data that was projected
    onto the same global variational distributions (beta,eta)

    If `x.bp` != `y.bp`, returned model `xy.bp` is set to None. All other
    attributes (except for the merged xi and eta) are inherited from `x`.

    Parameters
    ----------
    x : `scHPF`
    y : `scHPF`
        The scHPF instance whose rows in the output should be at the
        corresponding indices `y_ixs`
    y_ixs : ndarray
        Row indices of `y` in the returned distributions. Must be 1-d and
        have same number of rows as `y`, have no repeats, and have no index
        greater than or equal to x.ncells + y.ncells.


    Returns
    -------
    ab : `scHPF`

    """
    assert x.dp == y.dp
    assert x.eta == y.eta
    assert x.beta == y.beta

    xy = deepcopy(x)
    if y.bp != x.bp:
        xy.bp = None
    xy.xi = x.xi.combine(y.xi, y_ixs)
    xy.theta = x.theta.combine(y.theta, y_ixs)
    return xy


def run_trials(X, nfactors,
        ntrials=5,
        min_iter=30,
        max_iter=1000,
        check_freq=10,
        epsilon=0.001,
        better_than_n_ago=5,
        dtype=np.float64,
        verbose=True,
        vcells = None,
        vX = None,
        loss_function=None,
        model_kwargs = {},
        return_all = False,
        reproject = False,
        reproject_kwargs = {},
        batchsize=0,
        beta_theta_simultaneous=False,
        loss_smoothing=1
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
    ntrials : int,  optional (Default 5)
        Number of random initializations for training
    min_iter: int, optional (Default 30)
        Minimum number of interations for training.
    max_iter: int, optional (Default 1000):
        Maximum number of interations for training.
    check_freq: int, optional (Default 10)
        Number of training iterations between calculating loss.
    epsilon: float, optional (Default 0.001)
        Percent change of loss for convergence.
    better_than_n_ago: int, optional (Default 5)
        Stop condition if loss is getting worse.  Stops training if loss
        is worse than `better_than_n_ago`*`check_freq` training steps
        ago and getting worse.
    dtype : datatype, optional (Default np.float64)
        np.float64 or np.float32
    verbose: bool, optional (Default True)
        verbose
    vcells : coo_matrix, optional (Default None)
        cells to use in a validation loss function
    vX : coo_matrix, optional (Default None)
        nonzero entries from the cells in vX
    loss_function : function, optional (Default None)
        A loss function to asses convergence that accepts data, model
        variational parameters, and model hyperparameters.  Note this is
        distinct from the `loss_function` argument in scHPF._fit (called by
        scHPF.fit and scHPF.project), which assumes a fixed reference to data
        is included in the function and *does not* accept data as an argument.
    model_kwargs: dict, optional (Default {})
        dictionary of additional keyword arguments for model
        initialization
    return_all: bool, optional (Default False)
        return all models
    reproject: bool, optional (Default False)
        Reproject the data onto the frozen gene variables before calculating
        loss. The reprojected loss will be added to the end of loss as a
        sublist. Note that this reprojection will *not* use the `loss_function`
        argument, and instead use the default provided log likelihood
    reproject_kwargs: dict, optional (Default {'replace':True})
        Only used if `reproject` is True. Keyword args for scHPF.project.
        'replace':True cannot be changed, and will be overwritten if given
    batchsize: int, optional (Defualt 0)
        Number of cells to use per training round. All cells used if 0.
    loss_smoothing: int, optional (Default: 1)
        Smooth loss up to `loss_smoothing` check frequencies ago. 1 results in
        no smoothing. Intended to be used with batching when assessing
        convergence based on training loss, where a good value might be
        int(ncells/n_batches)

    Returns
    -------
    best_model: scHPF
        The model with the best loss facter `ntrials` random initializations
        and training runs
    rejected_models: list, optional
        Rejected models, ordered by decreasing loss . Only returned if
        return_all is True
    """
    ncells, ngenes = X.shape
    if ngenes >= 20000:
        msg = 'WARNING: you are running scHPF with {} genes,'.format(ngenes)
        msg += ' which is more than the ~20k protein coding genes in the'
        msg += ' human genome. We suggest running scHPF on protein-coding'
        msg += ' genes only.'
        print(msg)

    # get the loss function for any data
    if loss_function is None:
        loss_function = partial(ls.mean_negative_pois_llh,
                single_process=False)

    # check data we're using for loss
    if vcells is not None:
        assert X.shape[1] == vcells.shape[1]
    if vX is not None:
        assert vX.shape == X.shape
    else:
        vX = X
    # setup loss fnc w/data (will be overridden if vcells is not None)
    data_loss_function = ls.loss_function_for_data(loss_function, vX)
    # setup smoothed_loss if using batches

    # run trials
    best_loss, best_model, best_t = np.finfo(np.float64).max, None, None
    models, losses = [], [] # only used if return_all
    for t in range(ntrials):
        # make a new model
        model = scHPF(nfactors=nfactors,
                    min_iter=min_iter, max_iter=max_iter,
                    check_freq=check_freq, epsilon=epsilon,
                    better_than_n_ago=better_than_n_ago,
                    verbose=verbose, dtype=dtype,
                    **model_kwargs
                    )

        # override the loss function data if we have vcells
        # (must be redone for each new model)
        if vcells is not None:
            proj_kwargs = dict(reinit=False,
                               min_iter=1,
                               max_iter=min(10, check_freq),
                               check_freq= check_freq+1,
                               verbose=False
                               )
            data_loss_function = ls.projection_loss_function(
                    loss_function, vcells, nfactors,
                    proj_kwargs=proj_kwargs)
            def checkstep_function(**kwargs):
                loss = ls.loss_function_for_data(loss_function, X)
                print('\ttrain:', '{0:.6f}'.format(loss(**kwargs)))
        else:
            checkstep_function = None

        # fit the model
        model.fit(X, loss_function=data_loss_function,
                  checkstep_function=checkstep_function,
                  batchsize=batchsize, loss_smoothing=loss_smoothing,
                  beta_theta_simultaneous=beta_theta_simultaneous)
        if reproject:
            print('Reprojecting data...')
            reproject_kwargs['replace'] = True
            reproject_kwargs['reinit'] = False
            proj_loss = model.project(X, **reproject_kwargs)
            model.loss.append(proj_loss)
            loss = proj_loss[-1]
        else:
            loss = model.loss[-1]

        if loss < best_loss:
            best_model = model
            best_loss = loss
            best_t = t
            if verbose:
                print('New best!'.format(t))
        if return_all:
            models.append(model)
            losses.append(loss)
        if verbose:
            print('Trial {0} loss: {1:.6f}'.format(t, loss))
            print('Best loss: {0:.6f} (trial {1})'.format(best_loss, best_t))

    if return_all:
        return_order = np.argsort(losses)
        ordered_models = [models[i] for i in return_order]
        assert ordered_models[0] == best_model
        return best_model, ordered_models[1:]
    else:
        return best_model


# TODO deal with verbosity
def run_trials_pool(X, nfactors,
        ntrials=5,
        njobs=0,
        max_threads=None,
        min_iter=30,
        max_iter=1000,
        check_freq=10,
        epsilon=0.001,
        better_than_n_ago=5,
        dtype=np.float64,
        verbose=True,
        vcells = None,
        vX = None,
        loss_function=None,
        model_kwargs = {},
        return_all = False,
        reproject = False,
        reproject_kwargs = {},
        batchsize=0,
        beta_theta_simultaneous=False,
        loss_smoothing=1
        ):
    """
    Train with multiple random initializations, selecting model with best loss.
    Parallelization is done at the trial level rather than within computations

    As scHPF uses non-convex optimization, it benefits from training with
    multiple random initializations to avoid local minima.

    Parameters
    ----------
    X: coo_matrix
        Data to fit
    nfactors: int or list of ints
        Number of factors (K), may be a list for multiple k
    ntrials : int,  optional (Default 5)
        Number of random initializations for training
    njobs : int, optional (Default 0)
        Maximum number of threads in the threadpool.  0 will use all available.
    min_iter: int, optional (Default 30)
        Minimum number of interations for training.
    max_iter: int, optional (Default 1000):
        Maximum number of interations for training.
    check_freq: int, optional (Default 10)
        Number of training iterations between calculating loss.
    epsilon: float, optional (Default 0.001)
        Percent change of loss for convergence.
    better_than_n_ago: int, optional (Default 5)
        Stop condition if loss is getting worse.  Stops training if loss
        is worse than `better_than_n_ago`*`check_freq` training steps
        ago and getting worse.
    dtype : datatype, optional (Default np.float64)
        np.float64 or np.float32
    verbose: bool, optional (Default True)
        verbose
    vcells : coo_matrix, optional (Default None)
        cells to use in a validation loss function
    vX : coo_matrix, optional (Default None)
        nonzero entries from the cells in vX
    loss_function : function, optional (Default None)
        A loss function that accepts data, model variational parameters,
        and model hyperparameters.  Note this is distinct from the
        `loss_function` argument in scHPF._fit (called by scHPF.fit and
        scHPF.project), which assumes a fixed reference to data is included
        in the function and *does not* accept data as an argument.
    model_kwargs: dict, optional (Default {})
        dictionary of additional keyword arguments for model
        initialization
    return_all: bool, optional (Default False)
        return all models
    reproject: bool, optional (Default False)
        Reproject the data onto the frozen gene variables before calculating
        loss. The reprojected loss will be added to the end of loss as a
        sublist. Note that this reprojection will *not* use the `loss_function`
        argument, and instead use the default provided log likelihood
    reproject_kwargs: dict, optional (Default {'replace':True})
        Only used if `reproject` is True. Keyword args for scHPF.project.
        'replace':True cannot be changed, and will be overwritten if given
    batchsize: int, optional (Defualt 0)
            Number of cells to use per training round. All cells used if 0.
    loss_smoothing: int, optional (Default: 1)
        Smooth loss up to `loss_smoothing` check frequencies ago. 1 results in
        no smoothing. Intended to be used with batching when assessing
        convergence based on training loss, where a good value might be
        int(ncells/n_batches)


    Returns
    -------
    best_models: list(scHPF)
        The model with the best loss facter `ntrials` random initializations
        and training runs for each value in nfactors
    rejected_models: list(list(scHPF)), optional
        Rejected models, ordered by corresponding nfactors and then by
        decreasing loss . Only returned if return_all is True
    """
    ngenes = X.shape[1]
    if ngenes >= 20000:
        msg = 'WARNING: you are running scHPF with {} genes,'.format(ngenes)
        msg += ' which is more than the ~20k protein coding genes in the'
        msg += ' human genome. We suggest running scHPF on protein-coding'
        msg += ' genes only.'
        print(msg)

    # get the loss function for any data
    if loss_function is None:
        loss_function = partial(ls.mean_negative_pois_llh,
                single_process=True)

    # check data we're using for loss
    if vcells is not None:
        assert X.shape[1] == vcells.shape[1]
    if vX is not None:
        assert vX.shape == X.shape
    else:
        vX = X
    # setup loss fnc w/data (will be overridden if vcells is not None)
    data_loss_function = ls.loss_function_for_data(loss_function, vX)

    # only need to create once because will be copied to processes
    # override the loss function data if we have vcells
    # (must be redone for each new model)
    if vcells is not None:
        proj_kwargs = dict(reinit=False,
                            min_iter=1,
                            max_iter=min(10, check_freq),
                            check_freq= check_freq+1,
                            verbose=False
                            )
        data_loss_function = ls.projection_loss_function(
                loss_function, vcells, nfactors,
                proj_kwargs=proj_kwargs)


    # function to fit model
    def fit_model(nfactors):
        model = scHPF(nfactors=nfactors,
                    min_iter=min_iter, max_iter=max_iter,
                    check_freq=check_freq, epsilon=epsilon,
                    better_than_n_ago=better_than_n_ago,
                    verbose=False, dtype=dtype,
                    **model_kwargs
                    )
        # fit the model
        model.fit(X, loss_function=data_loss_function,
                  checkstep_function=None, single_process=True,
                  batchsize=batchsize, loss_smoothing=loss_smoothing)
        if reproject:
            # print('Reprojecting data...')
            reproject_kwargs['replace'] = True
            proj_loss = model.project(X, loss_function=data_loss_function,
                    **reproject_kwargs)
            model.loss.append(proj_loss)
        return model

    # get nfactors for every trial
    if isinstance(nfactors, int):
        nfactors = [nfactors]
    trial_nfactors = [t for trial_set in [[K]*ntrials for K in nfactors] \
            for t in trial_set]

    # set max processes if not given
    if njobs == 0: njobs = min(cpu_count(), len(trial_nfactors))

    # training
    with Parallel(n_jobs=njobs, verbose=10) as pool: # make the pool
        candidates = pool( delayed(fit_model)(K)  for K in trial_nfactors)

    # get the best model for every K
    ordered_best, ordered_reject = [], []
    for i,K in enumerate(nfactors):
        my_candidates = candidates[i*ntrials : (i+1)*ntrials]
        loss = [m.loss[-1][-1] if reproject else m.loss[-1] for m in
                my_candidates]
        # print(list(zip([m.nfactors for m in my_candidates],loss)))
        best_ix = np.argmin(loss)
        ordered_best.append(my_candidates[best_ix])
        ordered_reject.append([my_candidates[i] for i in np.argsort(loss)[1:]])
    if return_all:
        return ordered_best, ordered_reject
    else:
        return ordered_best
