#!/usr/bin/env python

import pytest


def test__setup(model, data):
    bp, dp, xi, eta, theta, beta = model._setup(X=data,
            freeze_genes=False, reinit=True)

