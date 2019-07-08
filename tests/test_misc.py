#!/usr/bin/env python

import schpf

def test_version():
    assert schpf.__version__ is not None
    assert schpf.__version__ == '0.2.5'
