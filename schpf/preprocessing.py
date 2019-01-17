#!/usr/bin/env python

import numpy as np
from scipy.sparse import coo_matrix

import pandas as pd

def load_coo(filename):
    """Load a sparse coo matrix

    Assumes first column (dense row ids) are cells, second column (dense
    column ids) are genes, and third column are nonzero counts. Also assumes
    row and column ids are 0-indexed.

    Parameters
    ----------
    filename : str
        file to load

    Returns
    -------
    coo : coo_matrix
    """
    raw = np.loadtxt(filename, delimiter='\t', dtype=int)
    sparse = coo_matrix((raw[:,2], (raw[:,0],raw[:,1])))
    return sparse


def load_loom(filename):
    """Load data from a loom file

    Parameters
    ----------
    filename: str
        file to load

    Returns
    -------
    coo : coo_matrix
        cell x gene sparse count matrix
    genes : Dataframe
        Dataframe of gene attributes.  Attributes are ordered so
        Accession and Gene are the first columns, if those attributs are
        present
    """
    import loompy
    # load the loom file
    with loompy.connect(filename) as ds:
        loom_genes = pd.DataFrame(dict(ds.ra.items()))
        loom_coo = ds.sparse().T

    # order gene attributes so Accession and Gene are the first two columns,
    # if they are present
    first_cols = []
    for colname in ['Accession', 'Gene']:
        if colname in loom_genes.columns:
            first_cols.append(colname)
    rest_cols = loom_genes.columns.difference(first_cols).tolist()
    loom_genes = loom_genes[first_cols + rest_cols]

    return loom_coo,loom_genes


def load_txt(filename,  ngene_cols=2):
    """Load data from a whitespace delimited txt file

    Parameters
    ----------
    filename : str
        file to load.  Expected to be a gene x cell whitespace-delimited file
        without a header where the first `ngene_cols` are gene identifiers,
        names or other metadata.
    ngene_cols : int, default 2
        The number of columns that contain row attributes (ie gene id/names)

    Returns
    -------
    coo : coo_matrix
        cell x gene sparse count matrix
    genes : pd.DataFrame
        ngenes x ngene_cols array of gene names/attributes
    """
    assert( ngene_cols > 0 )
    gene_cols = list(range(ngene_cols))

    if filename.endswith('.gz') or filename.endswith('.bz2'):
        msg = '......'
        msg+= 'WARNING: Input file {} is compressed. '.format(filename)
        msg+= 'It may be faster to manually decompress before loading.'
        print(msg)

        df = pd.read_csv(filename, header=None, memory_map=True,
                delim_whitespace=True)

        genes = df[gene_cols]
        dense = df.drop(columns=gene_cols).values.T
        nz = np.nonzero(dense)
        coo = coo_matrix((dense[nz], nz), shape=dense.shape, dtype=np.int32)
    else:
        genes, rows, cols, values = [], [], [], []

        # load row by row to conserve memory + actually often faster
        with open(filename) as f:
            # for each gene/row
            for g, l in enumerate(f):
                llist = l.split()
                genes.append(llist[:ngene_cols])
                r, c, val = [], [], []

                # for each cell/column
                for cell,v in enumerate(llist[ngene_cols:]):
                    if v != '0':
                        r.append(int(cell))
                        c.append(int(g))
                        val.append(int(v))

                rows.extend(r)
                cols.extend(c)
                values.extend(val)

            if (g%5000 == 0) and (g!=0):
                print('......loaded {} genes for {} cells'.format(g+1, cell+1))

        ncells, ngenes = len(llist[ngene_cols:]), g+1
        coo = coo_matrix((np.array(values), (np.array(rows),np.array(cols))),
                shape=(ncells,ngenes), dtype=np.int32)
        genes = pd.DataFrame(genes)

    return coo, genes


def min_cells_expressing_mask(counts, min_cells, verbose=True):
    """Get a mask for genes expressed by a minimum number of cells

    Parameters
    ----------
    counts : ndarray or coo_matrix
        A cell x gene coo_matrix of counts
    min_cells: numeric
        the minimum number (if int) or proportion (if float between 0 and 1)
        of cells in which we must observe transcripts of the gene for
        inclusion in the dataset.  If `min_cells` is between 0 and 1, sets
        the threshold to round(min_cells * ncells)
    verbose : bool, default True
        if True, print the number of cells when a numbr between 0 and 1 is given

    Returns
    -------
    passing_mask : ndarray
        boolean array of passing genes
    """
    if min_cells < 1 and min_cells > 0:
        min_cells_frac = min_cells
        min_cells = round(min_cells_frac * counts.shape[0])
        msg = '......requiring {}% of cells = {} cells observed expressing for'
        msg += ' gene inclusion'
        print(msg.format(100 * min_cells_frac, min_cells))
    return counts.astype(bool).sum(axis=0).A[0,:] >= min_cells


def genelist_mask(candidates, genelist, whitelist=True, split_on_dot=True):
    """Get a mask for genes on or off a list

    Parameters
    ----------
    candidates : pd.Series
        Candidate genes (from matrix)
    genelist : pd.Series
        List of genes to filter against
    whitelist : bool, default True
        Is the gene list a whitelist (True), where only genes on it should
        be kept or a blacklist (False) where all genes on it should be
        excluded
    split_on_dot : bool, default True
        If True, remove part of gene identifier after '.'.  We do this by
        default because ENSEMBL IDs contain version numbers after periods.

    Returns
    -------
    passing_mask : ndarray
        boolean array of passing genes
    """
    if split_on_dot:
        candidates = candidates.str.split('.').str[0]
        genelist = genelist.str.split('.').str[0]

    if whitelist:
        mask = candidates.isin(genelist)
    else:
        mask = ~candidates.isin(genelist)

    return mask.values
