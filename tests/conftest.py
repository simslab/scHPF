#!/usr/bin/env python

import numpy as np
from scipy.sparse import coo_matrix
import pytest
from schpf import scHPF

np.random.seed(42)

N_CELLS, N_GENES, NZ_FRAC, N_FACTORS = (300, 1000, 0.03, 4)
NNZ = int(N_CELLS * N_GENES * NZ_FRAC)

# Fixtures
@pytest.fixture()
def data():
    X_data = np.random.negative_binomial(2, 0.5, NNZ)
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
def model_uninit(request):
    model = scHPF(N_FACTORS, dtype=request.param)
    return model


@pytest.fixture()
def model(model_uninit, data):
    model_uninit._initialize(data)
    return model_uninit
