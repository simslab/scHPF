#!/usr/bin/env python

import numpy as np
from pathlib import Path

import pytest

from schpf import preprocessing as prep


TXT = str(Path(__file__).parent / \
        Path('_data/PJ030merge.c300t400_g0t500.matrix.txt'))
NCELLS = 100
NGENES = 500

# TODO figure out how to get this without going this far up tree or doubling
# perhaps make a small copy?
PROTEIN_CODING = Path(*Path(__file__).parts[-3]) / Path(
        'resources/gencode.v29.annotation.gene_l1l2.pc_TRC_IGC.stripped.txt')


def test_load_coo():
    assert False


def test_load_loom():
    assert False


@pytest.mark.parametrize('ngene_cols', [2,3])
def test_load_txt(ngene_cols):
    coo, genes = prep.load_txt(TXT, ngene_cols)
    assert genes.shape[1] == ngene_cols
    assert coo.shape[1] == NGENES
    assert genes.shape[0] == NGENES
    assert coo.shape[0]  == NCELLS + 2 - ngene_cols


def test_min_cells_expressing_mask():
    assert False


def test_genelist_mask():
    assert False


def test_load_and_filter():
    assert False
