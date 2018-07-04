#!/usr/bin/python

import numpy as np
import pandas as pd


def load_exp_matrix(matrix_file, gene_file=''):
    """ Read a gene expression matrix with no header and rows of the form:
        ENSEMBL_ID    GENE_SYMBOL    CELL_0    CELL_1    ...
        Return a gene x cell dataframe and a (ens, gene symbol) dataframe
    Parameters
    ----------
    matrix_file : str
        path to file with matrix
    Returns
    ----
    matrix: DataFrame
        pandas gene x cell dataframe
    genes: DataFrame
        gene x 2 dataframe of ensembl, gene pairs
    """
    df = pd.read_csv(matrix_file, sep='\t', header=None)
    if gene_file is not None and len(gene_file) > 0:
        genes = pd.read_csv(gene_file, header=None, sep='\t',
                names=['ens', 'gene'])
    else:
        genes = df[[0,1]].copy()
        genes.columns = ['ens', 'gene']
        del df[0], df[1]
        df.columns = np.arange(df.columns.size)

    df.index = genes['ens']
    genes.index = genes['ens']
    return df, genes


def write_exp_matrix(matrix, genes, outfile, outfile_gene=''):
    """
    Takes a gene x cell dataframe and a (ens, gene symbol) dataframe

    Parameters
    ----------
    matrix : pd dataframe
        the gene x cell matrix of expression values
    genes : pd dataframe
        the gene x 2 dataframe of ensmbl ids and gene names
    outfile : str
        where to write matrix file.  directory must exist.
    outfile_gene : str, optional
        where to write gene file.  not written if not given or len = 0 ['']

    Notes
    -----
    matrix and genes must have same index
    """
    pd.concat([genes, matrix], axis=1, join='inner').to_csv(outfile,
            header=False, index=False, sep='\t')

    if outfile_gene is not None and len(outfile_gene):
        genes.to_csv(outfile_gene, sep='\t', header=None, index=None)


def load_sparse_exp(coo_file, dtype=np.float32):
    """
    assume coo_file formatted like: cell_ix gene_ix count
    """
    raw = np.loadtxt(coo_file, dtype=dtype)
    indices, values = raw[:,:2].astype(int), raw[:,2]
    ncells, ngenes = (np.max(indices, axis=0) + 1).astype(int)
    nsamples = raw.shape[0]
    return nsamples, ncells, ngenes, indices, values


def load_genes(genefile):
    genes = pd.read_csv(genefile, header=None, delim_whitespace=True,
            names=['ens', 'gene'])
    return genes
