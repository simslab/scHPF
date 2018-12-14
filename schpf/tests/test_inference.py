#!/usr/bin/env python

import numpy as np
from scipy.sparse import coo_matrix
from scipy.special import logsumexp, digamma, gammaln

import pytest
from numpy.testing import assert_allclose

from schpf import hpf_numba, scHPF

# globals & seed
np.random.seed(42)

N_CELLS, N_GENES, NZ_FRAC, N_FACTORS = (300, 1200, 0.03, 4)
NNZ = int(N_CELLS * N_GENES * NZ_FRAC)

# Fixtures
@pytest.fixture()
def data():
    X_data = np.random.poisson(0.5, NNZ)
    X_data[X_data==0] = 1
    cell_ix = np.random.randint(0, N_CELLS, NNZ, dtype=np.int32)
    gene_ix = np.random.randint(0, N_GENES, NNZ, dtype=np.int32)
    X = coo_matrix(
            (X_data, (cell_ix, gene_ix)),
            (N_CELLS, N_GENES),
            dtype=np.int32)
    X.sum_duplicates()
    return X


# TODO make these actual unit tests by making distributions from scratch
@pytest.fixture(params=[np.float64, np.float32])
def model(request, data):
    model = scHPF(N_FACTORS, dtype=request.param)
    model._initialize(data)
    return model


@pytest.fixture()
def Xphi(data, model):
    random_phi = np.random.dirichlet( np.ones(N_FACTORS),
            data.data.shape[0]).astype(model.dtype)
    return data.data[:,None] * random_phi


# Tests

@pytest.mark.parametrize('x', [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_cython_digamma(x, dtype):
    x = dtype(x)
    # using approx_equal for float32 :(
    assert_allclose(hpf_numba.psi(x), digamma(x))


@pytest.mark.parametrize('x', [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_cython_gammaln(x, dtype):
    x = dtype(x)
    # using approx_equal for float32 :(
    assert_allclose(hpf_numba.cgammaln(x), gammaln(x))


def test_compute_Xphi_numba(data, model):
    def compute_Xphi_numpy(X, theta, beta):
        logrho = theta.e_logx[X.row, :] + beta.e_logx[X.col, :]
        logphi = logrho - logsumexp(logrho, axis=1)[:,None]
        return X.data[:,None] * np.exp(logphi)
    Xphi = compute_Xphi_numpy(data, model.theta, model.beta)
    # increase rtol for float32
    assert_allclose(
            hpf_numba.compute_Xphi_data(
                data.data, data.row, data.col,
                model.theta.vi_shape, model.theta.vi_rate,
                model.beta.vi_shape, model.beta.vi_rate),
            Xphi,
            rtol=1e-5, atol=0)


def test_compute_theta_shape_numba(model, Xphi, data):
    reference = np.zeros((N_CELLS, N_FACTORS), dtype=model.dtype)
    for k in range(N_FACTORS):
        reference[:,k] = coo_matrix(
                         (Xphi[:, k], (data.row, data.col)),
                         (N_CELLS, N_GENES)
                        ).sum(1).A[:,0]
    reference += model.a
    assert_allclose(
            hpf_numba.compute_loading_shape_update(
                Xphi, data.row, N_CELLS, model.a),
            reference)


def test_compute_beta_shape_numba(model, Xphi, data):
    reference = np.zeros((N_GENES, N_FACTORS), dtype=model.dtype)
    for k in range(N_FACTORS):
        reference[:,k] = coo_matrix(
                         (Xphi[:, k], (data.col, data.row)),
                         (N_GENES, N_CELLS)
                        ).sum(1).A[:,0]
    reference += model.c
    assert_allclose(
            hpf_numba.compute_loading_shape_update(
                Xphi, data.col, N_GENES, model.c),
            reference)


def test_compute_theta_rate_numba(model):
    reference = model.xi.e_x[:,None] + model.beta.e_x.sum(0)[None,:]
    assert_allclose(
            hpf_numba.compute_loading_rate_update(
                model.xi.vi_shape, model.xi.vi_rate,
                model.beta.vi_shape, model.beta.vi_rate),
            reference
            )


def test_compute_eta_rate_numba(model):
    reference = model.beta.e_x.sum(axis=1) + model.dp
    assert_allclose(
            hpf_numba.compute_capacity_rate_update(
                model.beta.vi_shape, model.beta.vi_rate,
                model.dp),
            reference,
            rtol=1e-6, atol=0)


def test_llh_pois(data, model):
    e_rate = model.theta.e_x @ model.beta.e_x.T
    desired = data.data * np.log(e_rate[data.row, data.col]) \
                - e_rate[data.row, data.col] \
                - gammaln(data.data + 1)
    assert_allclose(
            hpf_numba.compute_pois_llh(data.data, data.row, data.col,
                model.theta.vi_shape, model.theta.vi_rate,
                model.beta.vi_shape, model.beta.vi_rate),
            desired,
            rtol=1e-6, atol=0)


