#!/usr/bin/env python

from collections import namedtuple

import numpy as np
from scipy.stats import hypergeom


def max_pairwise(gene_scores, ntop=300):
    """ Get the maximum pairwise overlap of top genes

    Parameters
    ----------
    gene_scores : ndarray
        (ngenes, nfactors) array of gene scores
    ntop : int (optional, default 300)
        Number of top genes to consider in each factor

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
    max_pairwise = 0
    for i in range(tops.shape[1]):
        for j in range(tops.shape[1]):
            if i >= j:
                continue
            overlap = len(np.intersect1d(tops[:,i], tops[:,j]))
            if overlap > max_pairwise:
                max_pairwise = overlap

    p = hypergeom.pmf(k=max_pairwise, M=gene_scores.shape[0],
                N=ntop, n=ntop) \
        + hypergeom.sf(k=max_pairwise, M=gene_scores.shape[0],
                N=ntop, n=ntop)
    Overlap = namedtuple('Overlap', ['overlap', 'p'])
    return Overlap(max_pairwise, p)
