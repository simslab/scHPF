#!/usr/bin/env python

from collections import namedtuple
import numpy as np
from scipy.sparse import coo_matrix
from scipy.special import logsumexp, digamma, gammaln

import pytest
from numpy.testing import (
    assert_approx_equal,
    assert_allclose,
    assert_array_almost_equal
)

from schpf import hpf_numba, scHPF

# globals & seed
np.random.seed(42)

N_CELLS, N_GENES, NZ_FRAC, N_FACTORS = (300, 1200, 0.03, 4)
NNZ = int(N_CELLS * N_GENES * NZ_FRAC)

# dummy expression counts
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


@pytest.mark.parametrize('x', [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_cython_digamma(x, dtype):
    # xes = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    x = dtype(x)
    assert_approx_equal(hpf_numba.psi(x), digamma(x))


@pytest.mark.parametrize('x', [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_cython_gammaln(x, dtype):
    x = dtype(x)
    assert_approx_equal(hpf_numba.cgammaln(x), gammaln(x))


def test_compute_Xphi_numba(data, model):
    def compute_Xphi_numpy(X, theta, beta):
        logrho = theta.e_logx[X.row, :] + beta.e_logx[X.col, :]
        logphi = logrho - logsumexp(logrho, axis=1)[:,None]
        return X.data[:,None] * np.exp(logphi)
    # reference value
    Xphi = compute_Xphi_numpy(data, model.theta, model.beta)
    assert_array_almost_equal(
            hpf_numba.compute_Xphi_data(
                data.data, data.row, data.col,
                model.theta.vi_shape, model.theta.vi_rate,
                model.beta.vi_shape, model.beta.vi_rate),
            Xphi)


def test_compute_theta_shape_numba(model, Xphi, data):
    update = np.zeros((N_CELLS, N_FACTORS), dtype=model.dtype)
    for k in range(N_FACTORS):
        update[:,k] = coo_matrix(
                         (Xphi[:, k], (data.row, data.col)),
                         (N_CELLS, N_GENES)
                        ).sum(1).A[:,0]
    update += model.a
    assert_allclose(update,
            hpf_numba.compute_loading_shape_update(
                Xphi, data.row, N_CELLS, model.a),
            )


def test_compute_beta_shape_numba(model, Xphi, data):
    update = np.zeros((N_GENES, N_FACTORS), dtype=model.dtype)
    for k in range(N_FACTORS):
        update[:,k] = coo_matrix(
                         (Xphi[:, k], (data.col, data.row)),
                         (N_GENES, N_CELLS)
                        ).sum(1).A[:,0]
    update += model.c
    assert_allclose(
            hpf_numba.compute_loading_shape_update(
                Xphi, data.col, N_GENES, model.c),
            update)
