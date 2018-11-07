#!/usr/bin/env python

import numpy as np
from scipy.sparse import coo_matrix

# try:
    # import loompy
# except ImportError:
    # print('Warning: could not import loompy')

# import pandas as pd


def load_tab_delim(filename):
    pass


def load_loom(filename):
    pass


def load_coo(filename):
    raw = np.loadtxt(filename, delimiter='\t', dtype=int)
    sparse = coo_matrix((raw[:,2], (raw[:,0],raw[:,1])))
    return sparse


