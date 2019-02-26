#!/usr/bin/env python

import numpy as np

import pytest
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal

from schpf import HPF_Gamma, scHPF

"""For tests of inference, see test_inference.py
"""

def test__setup_meanvar(model_uninit, data):
    bp, dp, xi, eta, theta, beta = model_uninit._setup(X=data,
            freeze_genes=False, reinit=True)
    cell_sums = data.sum(axis=1)
    gene_sums = data.sum(axis=0)

    # test hyperparams set to mean/var ratios
    assert_equal(bp, np.mean(cell_sums) / np.var(cell_sums))
    assert_equal(dp, np.mean(gene_sums) / np.var(gene_sums))


def test__setup_dims(model_uninit, data):
    bp, dp, xi, eta, theta, beta = model_uninit._setup(X=data,
            freeze_genes=False, reinit=True)

    assert_equal(xi.vi_shape.shape[0], data.shape[0])
    assert_equal(xi.vi_rate.shape[0], data.shape[0])
    assert_equal(len(xi.vi_shape.shape), 1)
    assert_equal(len(xi.vi_rate.shape), 1)

    assert_equal(eta.vi_shape.shape[0], data.shape[1])
    assert_equal(eta.vi_rate.shape[0], data.shape[1])
    assert_equal(len(eta.vi_shape.shape), 1)
    assert_equal(len(eta.vi_rate.shape), 1)

    assert_equal(theta.vi_shape.shape[0], data.shape[0])
    assert_equal(theta.vi_rate.shape[0], data.shape[0])
    assert_equal(theta.vi_shape.shape[1], model_uninit.nfactors)
    assert_equal(theta.vi_rate.shape[1], model_uninit.nfactors)

    assert_equal(beta.vi_shape.shape[0], data.shape[1])
    assert_equal(beta.vi_rate.shape[0], data.shape[1])
    assert_equal(beta.vi_shape.shape[1], model_uninit.nfactors)
    assert_equal(beta.vi_rate.shape[1], model_uninit.nfactors)


@pytest.mark.parametrize('a_dims', [[5,], [5,10]])
@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_HPF_Gamma_combine(a_dims, dtype):
    a_vi_shape = np.ones(a_dims, dtype=dtype)
    a_vi_rate = np.ones(a_dims, dtype=dtype)
    a = HPF_Gamma(a_vi_shape, a_vi_rate)

    b_dims = a_dims.copy()
    b_dims[0] = 3
    b_vi_shape = 2*np.ones(b_dims, dtype=dtype)
    b_vi_rate = 2*np.ones(b_dims, dtype=dtype)
    b = HPF_Gamma(b_vi_shape, b_vi_rate)

    b_ix = [0,5,7]
    ab = a.combine(b, b_ix)
    assert_equal(ab.dims[0], a.dims[0] + b.dims[0])
    # check b rows
    assert_array_equal(ab.vi_shape[b_ix], b.vi_shape)
    assert_array_equal(ab.vi_rate[b_ix], b.vi_rate)
    # check a rows too
    a_ix = np.setdiff1d(np.arange(ab.dims[0]), b_ix)
    print(a_ix)
    assert_array_equal(ab.vi_shape[a_ix], a.vi_shape)
    assert_array_equal(ab.vi_rate[a_ix], a.vi_rate)

    b_ix = [4]
    with pytest.raises(AssertionError):
        ab = a.combine(b, b_ix)

    b_ix = [0,1,2,3]
    with pytest.raises(AssertionError):
        ab = a.combine(b, b_ix)

    b_ix = [0,1,2,2]
    with pytest.raises(AssertionError):
        ab = a.combine(b, b_ix)

    b_ix = [7,8,9]
    with pytest.raises(AssertionError):
        ab = a.combine(b, b_ix)


@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_project(data, dtype):
    b_idx = np.random.choice(data.shape[0], 10)
    a_idx = np.setdiff1d(np.arange(data.shape[0]), b_idx)
    data_csr = data.tocsr()
    a_data = data_csr[a_idx].tocoo()
    b_data = data_csr[b_idx].tocoo()

    a_model = scHPF(5, dtype=dtype)
    a_model._initialize(a_data)
    b_model = a_model.project(b_data)
    # check genes frozen
    assert_array_equal(b_model.eta.vi_shape, a_model.eta.vi_shape)
    assert_array_equal(b_model.eta.vi_rate, a_model.eta.vi_rate)
    assert_array_equal(b_model.beta.vi_shape, a_model.beta.vi_shape)
    assert_array_equal(b_model.beta.vi_rate, a_model.beta.vi_rate)
    # check cells different
    assert_equal(a_model.ncells, a_data.shape[0])
    assert_equal(b_model.ncells, b_data.shape[0])

# def test_run_trials(data):
    # pass
