#!/usr/bin/env python

import warnings
import numpy as np
from scipy.sparse import coo_matrix
import pandas as pd

from schpf.util import split_coo_rows


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
        msg = '.....'
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

                if ((g+1)%10000 == 0) and (g!=0):
                    print('\tloaded {} genes for {} cells'.format(
                        g+1, cell+1))

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

    TODO verbose option + return min_cells
    """
    if min_cells < 1 and min_cells > 0:
        min_cells_frac = min_cells
        min_cells = round(min_cells_frac * counts.shape[0])
        msg = '.....requiring {}% of cells = {} cells observed expressing for'
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


def choose_validation_cells(choices, nselect, group_ids=None, max_group_frac=0.5):
    """Randomly select cells, potentially accounting for groups

    Parameters
    ----------
    choices : ndarray or int
        Indices of cells to choose from.  If int is give, indices assumend
        to be np.arange(`choices`)
    nselect : int
        number of indices to return
    group_ids : ndarray, optional
        Group ids of cells.  len(`group_ids`) must == `choices` if `choices`
        is an int, and == len(`choices`) otherwise.  If `group_ids` is given,
        selected cells will be distributed approximately evenly over the
        labels under the constraint that at most floor(group_size *
        `max_group_frac`) can be selected from a group.
    max_group_frac : float, optional (default: 0.5)
        If `group_ids` given, the maximum fraction of cells in a group that
        can be selected.

    Returns
    -------
    selected_ix : ndarray
        1d array of selected ids (sorted).
    """
    if isinstance(choices, int):
        choices = np.arange(choices)

    if group_ids is None:
        return np.sort(np.random.choice(choices, nselect, replace=False))
    else:
        assert len(group_ids) == len(choices)

        label, remaining = np.unique(group_ids, return_counts=True)
        constraint = np.floor( remaining * max_group_frac ).astype(int)

        selected, n_remain = [], nselect
        # while unconstrained cells left and more requested
        while np.sum(constraint) > 0 and n_remain > 0:
            # calculate goals given remaining cells to select and
            # unconstrained cells left
            weights = (constraint > 0) / (constraint > 0).sum()
            goal_floor = np.floor(weights * n_remain).astype(int)
            remainder = np.sum(np.ceil(weights * n_remain) - goal_floor
                    ).astype(int)
            goal = goal_floor + np.random.multinomial(remainder, weights)
            # for each group
            for i in range(len(remaining)):
                # if there are unconstrained cells left in the group
                if constraint[i] > 0:
                    my_nchoose = min(goal[i], constraint[i])
                    my_choices = np.setdiff1d(choices[group_ids == label[i]],
                            selected)
                    # select the cells
                    chosen = np.random.choice(my_choices, my_nchoose,
                            replace=False)
                    selected.extend(list(chosen))
                    # update constraint
                    constraint[i] -= my_nchoose
                    n_remain -= my_nchoose
        if n_remain > 0:
            msg = "Could not select {} cells".format(nselect)
            msg += " with given group_ids under constraint max_group_frac"
            msg += "={}. {} cells selected.".format(max_group_frac, n_remain)
            warnings.warn(msg, UserWarning)

        return np.sort(selected)


def split_validation_cells(X, nselect, group_id_file='', max_group_frac=0.5,
        verbose=True):
    """ Split train and validation cells, potentially accounting from groups

    Parameters
    ----------
    X : coo_matrix
        Matrix to select validation cells from
    nselect : int
        Number of cells to select
    group_id_file : str, optional
        File containing group ids.  Should be loadable with np.loadtxt
    max_group_frac : float, optional (default: 0.5)
        If `group_id_file` given, the maximum fraction of cells in a group that
        can be selected.
    verbose : bool, optional (default: True)
        Verbose output

    Returns
    -------
    Xtrain : coo_matrix
        X with validation rows removed
    Xvalidation : coo_matrix
        Selected rows from X
    validation_ix : ndarray
        Indexes of selected rows in the intput matrix `X`
    """
    # load groups
    if group_id_file is not None and len(group_id_file):
        group_ids = np.loadtxt(group_id_file)
    else:
        group_ids = None

    # select cells
    selected_ids = choose_validation_cells(X.shape[0], nselect, group_ids)

    # write a message
    if verbose:
        ncells = len(selected_ids)
        msg = '.....{} cells selected'.format(ncells)
        if group_ids is not None:
            msg += ' ~~evenly from groups in {}'.format(group_id_file)
            msg += ' under constraint max_group_frac={}'.format(max_group_frac)
            msg += '\n\tGroup counts:'
            ids, id_counts = np.unique(group_ids[selected_ids],
                    return_counts=True)
            for i, c in zip(ids, id_counts):
                msg += '\n\t\t[{}] {}'.format(i, c)
        print(msg)

    # split cells
    Xvalidation, Xtrain =  split_coo_rows(X, selected_ids)
    return Xtrain, Xvalidation, selected_ids


def load_and_filter(infile, min_cells, whitelist='', blacklist='',
        filter_by_gene_name=False, no_split_on_dot=False, verbose=True):
    """ Composite of loading and filtering intended for use by CLI
    Parameters
    ----------
    infile : str
        Input data. Currently accepts either: (1) a whitespace-delimited gene
        by cell UMI count matrix with 2 leading columns of gene attributes
        (ENSEMBL_ID and GENE_NAME respectively), or (2) a loom file with at
        least one of the row attributes `Accession` or `Gene`, where `Accession`
        is an ENSEMBL id and `Gene` is the name.
    min_cells : float or int
        Minimum number of cells in which we must observe at least one transcript
        of a gene for the gene to pass filtering. If 0 <`min_cells`< 1, sets
        threshold to be `min_cells` * ncells, rounded to the nearest integer.
    whitelist : str, optional
        Tab-delimited file where first column contains ENSEMBL gene ids to
        accept, and second column contains corresponding gene names. If given,
        genes not on the whitelist are filtered from the input matrix.
        Superseded by blacklist. Default None.
    blacklist : str, optional
        Tab-delimited file where first column contains ENSEMBL gene ids to
        exclude, and second column is the corresponding gene name. Only
        performed if file given. Genes on the blacklist are excluded even if
        they are also on the whitelist.
    filter_by_gene_name : bool, optional
        Use gene name rather than ENSEMBL id to filter (with whitelist or
        blacklist).  Useful for datasets where only gene symbols are given.
        Applies to both whitelist and blacklist. Used by default when input
        is a loom file. Default False.
    no_split_on_dot : bool, optional
        Don't split gene symbol or name on period before filtering white and
        blacklist. We do this by default for ENSEMBL ids. Default False.
    verbose : bool, optional
        Print progress messages. Default True

    Returns
    -------
    filtered : ndarray
    genes : pd.DataFrame

    Raises
    ------
    ValueError
    """
    if verbose:
        print('Loading data.....')

    if infile.endswith('.loom'):
        umis, genes = load_loom(args.input)
        if 'Accession' in genes.columns:
            candidate_names = genes['Accession']
            genelist_col = 0
        elif 'Gene' in genes.columns:
            candidate_names = genes['Gene']
            genelist_col = 1
        else:
            msg = 'loom files must have at least one of the row '
            msg+= 'attributes: `Gene` or `Accession`.'
            raise ValueError(msg)
    else:
        umis, genes = load_txt(infile)
        genelist_col = 1 if filter_by_gene_name else 0
        candidate_names = genes[genelist_col]
    ncells, ngenes = umis.shape
    if verbose:
        print('.....found {} cells and {} genes'.format(ncells, ngenes))
        print('Generating masks for filtering.....')

    if min_cells < 0:
        raise ValueError('min_cells must be >= 0')
    mask = min_cells_expressing_mask(umis, min_cells)
    if whitelist is not None and len(whitelist):
        whitelist = pd.read_csv(whitelist, delim_whitespace=True, header=None)
        mask &= genelist_mask(candidate_names, whitelist[genelist_col],
                              split_on_dot = ~no_split_on_dot)
    if blacklist is not None and len(blacklist):
        blacklist = pd.read_csv(blacklist, delim_whitespace=True, header=None)
        mask &= genelist_mask(candidate_names, blacklist[genelist_col],
                              whitelist=False, split_on_dot = ~no_split_on_dot)

    if verbose:
        print('Filtering data.....')
    genes = genes.loc[mask]
    filtered = umis.tolil()[:,mask].tocoo() # must convert to apply mask

    return filtered, genes

