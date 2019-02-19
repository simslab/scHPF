#!/usr/bin/env python

from collections import namedtuple

import numpy as np
from scipy.stats import hypergeom
import pandas as pd


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
