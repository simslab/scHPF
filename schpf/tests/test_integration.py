#!/usr/bin/env python

import numpy as np

import pytest
from numpy.testing import assert_equal


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


# def test_run_trials(data):
    # pass
