#!/usr/bin/env python

from collections import namedtuple

import numpy as np
from scipy.stats import hypergeom


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

    overlap = last_max if second_greatest else max_pairwise
    p = hypergeom.pmf(k=overlap, M=gene_scores.shape[0],
                N=ntop, n=ntop) \
        + hypergeom.sf(k=overlap, M=gene_scores.shape[0],
                N=ntop, n=ntop)
    Overlap = namedtuple('Overlap', ['overlap', 'p'])
    return Overlap(overlap, p)
