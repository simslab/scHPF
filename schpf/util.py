#!/usr/bin/env python

from collections import namedtuple

import numpy as np
from scipy.stats import hypergeom
from scipy.sparse import csr_matrix
import pandas as pd


def mean_cellscore_fraction(cell_scores, ntop_factors=1):
    """ Get number of cells with a percentage of their total scores
    on a small number of factors

    Parameters
    ----------
    cell_scores : ndarray
        (ncells, nfactors) array of cell scores
    ntop_factors : int, optional (Default: 1)
        number of factors that can count towards domance

    Returns
    -------
    mean_cellscore_fraction : float
        The mean fraction of cells' scores that are contained within
        their top `ntop_factors` highest scoring factors

    """
    totals = np.sum(cell_scores, axis=1)
    ntop_scores = np.sort(cell_scores,axis=1)[:, -ntop_factors:]
    domsum = np.sum(ntop_scores, axis=1)
    domfrac = domsum/totals
    return np.mean(domfrac)


def mean_cellscore_fraction_list(cell_scores):
    """ Make a list of the mean dominant fraction at all possible numbers
        of ntop_factors
    """
    return [mean_cellscore_fraction(cell_scores, i+1)
                for i in range(cell_scores.shape[1])]


def max_pairwise(gene_scores, ntop=200, second_greatest=False):
    """ Get the maximum pairwise overlap of top genes

    Parameters
    ----------
    gene_scores : ndarray
        (ngenes, nfactors) array of gene scores
    ntop : int (optional, default 200)
        Number of top genes to consider in each factor
    second_greatest : bool, optional
        Return the second greatest pairwise overlap of top genes

    Returns
    -------
    max_pairwise : int
        The maximum pairwise overlap of the `ntop` highest scoring genes in
        each factors
    p : float
        Hypergeometric p value of max_pairwise, where the number of genes is
        the population size, `ntop` is the number of potential successes and
        the number of draws, and max_pairwise is the number of successes.
    """
    tops = np.argsort(gene_scores, axis=0)[-ntop:]
    max_pairwise, last_max = 0, 0
    for i in range(tops.shape[1]):
        for j in range(tops.shape[1]):
            if i >= j:
                continue
            overlap = len(np.intersect1d(tops[:,i], tops[:,j]))
            if overlap > max_pairwise:
                last_max = max_pairwise
                max_pairwise = overlap
            elif overlap > last_max:
                last_max = overlap

    overlap = last_max if second_greatest else max_pairwise
    p = hypergeom.pmf(k=overlap, M=gene_scores.shape[0],
                N=ntop, n=ntop) \
        + hypergeom.sf(k=overlap, M=gene_scores.shape[0],
                N=ntop, n=ntop)
    Overlap = namedtuple('Overlap', ['overlap', 'p'])
    return Overlap(overlap, p)


def max_pairwise_table(gene_scores, ntop_list=[50,100,150,200,250,300]):
    """ Get the maximum pairwise overlap at

    Parameters
    ----------
    gene_scores : ndarray
        (ngenes, nfactors) array of gene scores
    ntop_list : list, optional
        List of values of ntop to evaluate

    Returns
    -------
    df : DataFrame
    """
    max_overlap, p_max, max2_overlap, p_max2 = [],[],[],[]
    for ntop in ntop_list:
        o = max_pairwise(gene_scores, ntop, False)
        max_overlap.append( o.overlap )
        p_max.append( o.p )

        o2 = max_pairwise(gene_scores, ntop, True)
        max2_overlap.append( o2.overlap )
        p_max2.append( o2.p )
    df = pd.DataFrame({'ntop' : ntop_list, 'max_overlap' : max_overlap,
        'p_max' : p_max, 'max2_overlap' : max2_overlap, 'p_max2' : p_max2})
    return df


def split_coo_rows(X, split_indices):
    """Split a coo matrix into two

    Parameters
    ----------
    X : coo_matrix
        Matrix to split into two by row
    split_indices : ndarray
        Indices to use for the split.

    Returns
    -------
    a : coo_matrix
        rows from X specified in split_indices
    b : coo_matrix
        rows from X *not* specified in split_indices

    """
    a_indices = split_indices
    b_indices = np.setdiff1d(np.arange(X.shape[0]), split_indices)

    X_csr = X.tocsr()
    a = X_csr[a_indices, :].tocoo()
    b = X_csr[b_indices, :].tocoo()
    return a, b


def collapse_coo_rows(coo):
    """Collapse the empty rows of a coo_matrix

    Parameters
    ----------
    coo : coo_matrix
        Input coo_matrix which may have empty rows


    Returns
    -------
    collapsed_coo : coo_matrix
        coo with row indices adjusted to removed empty rows
    collapsed_indices : ndarray
        Indices of the returned rows in the original input matrix
    """
    nz_idx = np.where(coo.getnnz(1) > 0)[0]
    return coo.tocsr()[nz_idx].tocoo(), nz_idx


def insert_coo_rows(a, b, b_indices):
    """Insert rows from b into a at specified row indeces

    Parameters
    ----------
    a : sparse matrix
    b : sparse matrix
    b_indices : ndarray
        Indices in final matrix where b's rows should be. np.max(`b_indices`)
        must be a valid row index in the merged matrix with shape[0] =
        a.shape[0] + b.shape[0].  Must me ordered and unique.

    Returns
    -------
    ab :
        coo_matrix with rows re-indexed to have rows from b
    """
    # check arguments
    if a.shape[1] != b.shape[1]:
        msg = 'a.shape[1] must equal b.shape[1], received a with shape'
        msg += ' {} and b with shape {}'.format(a.shape, b.shape)
        raise ValueError(msg)
    if np.max(b_indices) >= a.shape[0] + b.shape[0]:
        msg = 'Invalid row indices {} for array with '.format(b_indices)
        msg += 'a.shape[0] + b.shape[0] = {} '.format(a.shape[0])
        msg += '+ {} = {}'.format(b.shape[0], a.shape[0]+b.shape[0])
        raise ValueError(msg)
    if not np.all(np.diff(b_indices) > 0):
        msg = '`b_indices` must be ordered without repeats. Received '
        msg += '{}'.format(b_indices)
        raise ValueError(msg)

    out_shape = (a.shape[0] + b.shape[0], a.shape[1])
    a = a.tocsr()
    b = b.tocsr()

    a_row, b_row = 0, 0
    data, indices, indptr = [], [], [0]
    for ab_row in range(out_shape[0]):
        if b_row < len(b_indices) and ab_row == b_indices[b_row]:
            my_row = b[b_row, :]
            b_row += 1
        else:
            my_row = a[a_row, :]
            a_row += 1
        data.append(my_row.data)
        indices.append(my_row.indices)
        indptr.append(indptr[-1] + my_row.indptr[1])

    ab = csr_matrix(
            (np.hstack(data), np.hstack(indices), np.array(indptr)),
            out_shape).tocoo()
    return ab


def minibatch_ix_generator(ncells, batchsize):
    assert ncells >= batchsize # allow equalitiy for testing
    ixs = np.arange(ncells)
    np.random.shuffle(ixs)
    start = 0
    while True:
        stop = start + batchsize
        if stop > ncells:
            stop = stop % ncells
            res = np.hstack([ixs[start:ncells], ixs[0:stop]])
        else:
            res = ixs[start:stop]
        start = stop % ncells # need mod for case where ncells=batchsize
        yield res
