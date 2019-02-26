#!/usr/bin/env python

import numpy as np

import pytest
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal

from schpf import HPF_Gamma, scHPF, combine_across_cells

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


def test__setup_freeze(model, data):
    my_data = data.tocsr()[:20].tocoo()
    bp, dp, xi, eta, theta, beta = (model.bp, model.dp, model.xi,
            model.eta, model.theta, model.beta)

    model.bp = None
    bp2, dp2, xi2, eta2, theta2, beta2 = model._setup(X=my_data,
            freeze_genes=True, reinit=True)

    # cell-side vals (should be the smae)
    assert_equal(dp2, dp)
    assert_equal(eta2, eta)
    assert_equal(beta2, beta)

    # gene-side vals
    assert bp2  != bp
    assert xi2.dims != xi.dims
    assert theta2.dims != theta.dims

    # check bp not updated w/freeze_genes if already set
    model.bp = bp
    bp3, _, _, _, _, _ = model._setup(X=my_data, freeze_genes=True, reinit=True)
    print(bp, bp2, bp3)
    assert bp3 == bp
    assert bp3 != bp2


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
    # get b indices
    b_idx = np.random.choice(data.shape[0], 10)
    # get remaining indices (for a)
    a_idx = np.setdiff1d(np.arange(data.shape[0]), b_idx)
    # split data
    data_csr = data.tocsr()
    a_data = data_csr[a_idx].tocoo()
    b_data = data_csr[b_idx].tocoo()

    # setup model for a_data
    a_model = scHPF(5, dtype=dtype)
    a_model._initialize(a_data)

    #project b_model
    b_model = a_model.project(b_data)
    # check genes frozen
    assert_equal(b_model.eta, a_model.eta)
    assert_equal(b_model.beta, a_model.beta)
    # check cells different
    assert_equal(a_model.ncells, a_data.shape[0])
    assert_equal(b_model.ncells, b_data.shape[0])


@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_combine_across_cells(data, dtype):
    # get b indices
    b_ixs = np.random.choice(data.shape[0], 10)
    # get a indices (remaining)
    a_ixs = np.setdiff1d(np.arange(data.shape[0]), b_ixs)
    # split data
    data_csr = data.tocsr()
    a_data = data_csr[a_ixs].tocoo()
    b_data = data_csr[b_ixs].tocoo()

    # setup model for a_data
    a = scHPF(5, dtype=dtype)
    a._initialize(a_data)
    # setup model for b_data w/same dp, eta, beta
    b = scHPF(5, dtype=dtype, dp=a.dp, eta=a.eta, beta=a.beta)
    b._initialize(b_data)

    ab = combine_across_cells(a, b, b_ixs)

    # check bp is None since it is different across the two models
    assert_equal(ab.bp, None)

    # check a locals where they should be in xi and eta
    assert_array_equal(ab.xi.vi_shape[a_ixs], a.xi.vi_shape)
    assert_array_equal(ab.xi.vi_rate[a_ixs], a.xi.vi_rate)
    assert_array_equal(ab.theta.vi_shape[a_ixs], a.theta.vi_shape)
    assert_array_equal(ab.theta.vi_rate[a_ixs], a.theta.vi_rate)

    # check b locals where they should be in xi and eta
    assert_array_equal(ab.xi.vi_shape[b_ixs], b.xi.vi_shape)
    assert_array_equal(ab.xi.vi_rate[b_ixs], b.xi.vi_rate)
    assert_array_equal(ab.theta.vi_shape[b_ixs], b.theta.vi_shape)
    assert_array_equal(ab.theta.vi_rate[b_ixs], b.theta.vi_rate)

    # check globals unchanged
    assert_equal(ab.eta, a.eta)
    assert_equal(ab.eta, b.eta)
    assert_equal(ab.beta, a.beta)
    assert_equal(ab.beta, b.beta)


# def test_run_trials(data):
    # pass
