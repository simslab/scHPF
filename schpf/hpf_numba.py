#!/usr/bin/python

import ctypes
import numpy as np
from scipy.sparse import coo_matrix
import numba
from numba.extending import get_cython_function_address as getaddr

# get numba-compatible digamma/psi and gammaln
# psi/digamma
psi_fnaddr = getaddr("scipy.special.cython_special", "__pyx_fuse_1psi")
psi_ftype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
psi = psi_ftype(psi_fnaddr)
# gammaln
gammaln_fnaddr = getaddr("scipy.special.cython_special", "gammaln")
gammaln_ftype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
cgammaln = gammaln_ftype(gammaln_fnaddr)

@numba.njit(parallel=True, nogil=True, fastmath=True)
def compute_pois_llh(X_data, X_row, X_col,
                     theta_vi_shape, theta_vi_rate,
                     beta_vi_shape, beta_vi_rate):
    ncells, ngenes = (theta_vi_shape.shape[0], beta_vi_shape.shape[0])
    nfactors, nnz = (theta_vi_shape.shape[1], X_data.shape[0])
    dtype = theta_vi_shape.dtype

    # precompute expectations
    theta_e_x = np.zeros_like(theta_vi_shape, dtype=dtype)
    for i in numba.prange(ncells):
        for k in range(nfactors):
            theta_e_x[i,k] = theta_vi_shape[i,k] / theta_vi_rate[i,k]

    beta_e_x = np.zeros_like(beta_vi_shape, dtype=dtype)
    for i in numba.prange(ngenes):
        for k in range(nfactors):
            beta_e_x[i,k] = beta_vi_shape[i,k] / beta_vi_rate[i,k]

    # compute llh
    llh = np.zeros(X_data.shape, dtype=dtype)
    for i in numba.prange(nnz):
        e_rate = np.zeros(1, dtype=dtype)[0]
        for k in range(nfactors):
            e_rate += theta_e_x[X_row[i],k] * beta_e_x[X_col[i], k]
        llh[i] = X_data[i] * np.log(e_rate) - e_rate \
            - cgammaln(X_data[i] + 1.0)
    return llh


def compute_pois_llh_sthread(X_data, X_row, X_col,
                             theta_vi_shape, theta_vi_rate,
                             beta_vi_shape, beta_vi_rate):
    """ Single-threaded version of compute_pois_llh"""
    raise NotImplementedError()


@numba.njit(parallel=True, nogil=True)
def compute_Xphi_data(X_data, X_row, X_col,
                     theta_vi_shape, theta_vi_rate,
                     beta_vi_shape, beta_vi_rate):
    """ Fast version of Xphi computation using numba & gsl_digamma

    Parameters
    ----------
    X_data : ndarray of np.int32
        (number_nonzero, ) array of nonzero values
    X_row : ndarray of np.int32
        (number_nonzero, ) array of row ids for each nonzero value
    X_col : ndarray (np.int32)
        (number_nonzero, ) array of column ids for each nonzero value
    theta_vi_shape : ndarray
        (ncells, nfactors) array of values for theta's variational shape
    theta_vi_rate : ndarray
        (ncells, nfactors) array of values for theta's variational rate
    beta_vi_shape : ndarray
        (ngenes, nfactors) array of values for beta's variational shape
    beta_vi_rate : ndarray
        (ngenes, nfactors) array of values for beta's variational rate
    """
    # convenience
    ncells, ngenes = (theta_vi_shape.shape[0], beta_vi_shape.shape[0])
    nfactors, nnz = (theta_vi_shape.shape[1], X_data.shape[0])
    dtype = theta_vi_shape.dtype

    # precompute theta.e_logx
    theta_e_logx = np.zeros_like(theta_vi_shape, dtype=dtype)
    for i in numba.prange(ncells):
        for k in range(nfactors):
            theta_e_logx[i,k] = psi(theta_vi_shape[i,k]) \
                                - np.log(theta_vi_rate[i,k])

    # precompute beta.e_logx
    beta_e_logx = np.zeros_like(beta_vi_shape, dtype=dtype)
    for i in numba.prange(ngenes):
        for k in range(nfactors):
            beta_e_logx[i,k] = psi(beta_vi_shape[i,k]) \
                               - np.log(beta_vi_rate[i,k])

    # compute Xphi
    Xphi = np.zeros((X_row.shape[0], theta_e_logx.shape[1]), dtype=dtype)
    for i in numba.prange(nnz):
        logrho = np.zeros((Xphi.shape[1]), dtype=dtype)
        for k in range(nfactors):
            logrho[k] = theta_e_logx[X_row[i],k] + beta_e_logx[X_col[i], k]

        #log normalizer trick
        rho_shift = np.zeros((Xphi.shape[1]), dtype=dtype)
        normalizer = np.zeros(1, dtype=dtype)[0]
        largest_in = np.max(logrho)
        for k in range(nfactors):
            rho_shift[k] = np.exp(logrho[k] - largest_in)
            normalizer += rho_shift[k]

        for k in range(nfactors):
            Xphi[i,k] = X_data[i] * rho_shift[k] / normalizer

    return Xphi


def compute_Xphi_data_sthread(X_data, X_row, X_col, theta_vi_shape,
        theta_vi_rate, beta_vi_shape, beta_vi_rate):
    """Single-threaded version of compute_Xphi_data
    """
    raise NotImplementedError()


@numba.njit(fastmath=True) #results unstable with prange. don't do it.
def compute_loading_shape_update(Xphi_data, X_keep, nkeep, shape_prior):
    """Compute gamma shape updates for theta or beta using numba

    Parameters
    ----------
    Xphi_data : ndarray
        (number_nonzero, nfactors) array of X * phi
    X_keep : ndarray
        (number_nonzer,) vector of indices along the axis of interest.
        If X is an (ncell,ngene) coo_matrix, this should be X.row when
        computing updates for theta and X.col when computing updates for
        beta
    nkeep : int
        Number of items on the axis of interest.  ncells when computing
        updates for theta, and ngenes for updates for beta
    shape_prior : float
        Hyperprior for parameter. a for theta, c for beta.

    """
    nnz, nfactors = Xphi_data.shape
    dtype = Xphi_data.dtype

    result = shape_prior * np.ones((nkeep, nfactors), dtype=dtype)
    for i in range(nnz):
        ikeep = X_keep[i]
        for k in range(nfactors):
            result[ikeep, k] += Xphi_data[i,k]
    return result


@numba.njit(fastmath=True)
def compute_loading_rate_update(prior_vi_shape, prior_vi_rate,
        other_loading_vi_shape, other_loading_vi_rate,):
    # shorter names
    pvs, pvr = (prior_vi_shape, prior_vi_rate)
    olvs, olvr = (other_loading_vi_shape, other_loading_vi_rate)
    dtype = prior_vi_shape.dtype

    other_loading_e_x_sum = np.zeros((olvs.shape[1]), dtype=dtype)
    for i in range(olvs.shape[0]):
        for k in range(olvs.shape[1]):
            other_loading_e_x_sum[k] += olvs[i,k] / olvr[i,k]

    result = np.zeros((pvs.shape[0], olvs.shape[1]), dtype=dtype)
    for i in range(pvs.shape[0]):
        prior_e_x = pvs[i] / pvr[i]
        for k in range(olvs.shape[1]):
            result[i, k] = prior_e_x + other_loading_e_x_sum[k]
    return result


@numba.njit(fastmath=True)
def compute_capacity_rate_update(loading_vi_shape, loading_vi_rate, prior_rate):
    dtype = loading_vi_shape.dtype
    result = prior_rate * np.ones((loading_vi_shape.shape[0],),
            dtype=dtype)
    for k in range(loading_vi_shape.shape[1]):
        for i in range(loading_vi_shape.shape[0]):
            result[i] += loading_vi_shape[i,k] / loading_vi_rate[i,k]
    return result
