#!/usr/bin/python

"""
Preprocess datasets for HPF.
"""

import os
import gc
import argparse

import yaml
import numpy as np
import pandas as pd

import scio


def filter_min_cells_expressing(df, genes, min_cells):
    """ Filter genes using the number of cells in which we observe transcripts
    Parameters
    ----------
    df : pandas dataframe
        (gene,cell) dataframe
    genes: pandas dataframe
        gene x 2 dataframe with rows (ens, gene symbol)
    min_cells: cells
        the minimum number of cells in which we must observe transcripts of the
        gene for inclusion in the dataset.  If `min_cells` is between 0 and 1, sets
        the threshold to round(min_cells * ncells)

    Returns
    -------
    passing_df : dataframe
        expression dataframe with only genes with >= min_cells
    passing_genes : dataframe
        genes that passed min_cells check
    """
    if min_cells < 1 and min_cells > 0:
        min_cells_frac = min_cells
        min_cells = round(min_cells_frac * df.shape[1])
        msg = 'Requiring {}% of cells = {} cells'
        print(msg.format(100 * min_cells_frac, min_cells))
    passing = df.astype(bool).sum(axis=1) >= min_cells
    return df.loc[passing], genes.loc[passing]


def filter_whitelist(df, genes, whitelist, whitelist_key='ens'):
    """ Filter genes based on a list of accepted genes in a file
    Parameters
    ----------
    df : pandas dataframe
        (gene,cell) dataframe
    genes: pandas dataframe
        gene x 2 dataframe with rows (ens, gene symbol)
    whitelist : str
        path to file with ens and gene symbol column for whitelisted genes
    whitelist_key : str
        name of column to use in whitelist

    Returns
    -------
    df_filt : dataframe
        expression dataframe with only accepted genes
    genes_filt : dataframe
        genes remaining after filter
    """
    accept = scio.load_genes(whitelist)
    accept[whitelist_key] = accept[whitelist_key].str.split('.').str[0]
    mask = genes[whitelist_key].str.split('.').str[0].isin(
            accept[whitelist_key].values)
    df_filt, genes_filt = df.loc[mask], genes.loc[mask]
    return df_filt, genes_filt


def filter_blacklist(df, genes, blacklist, blacklist_key='ens'):
    """ Remove genes in the blacklist file
    Parameters
    ----------
    df : pandas dataframe
        (gene,cell) dataframe
    genes: pandas dataframe
        gene x 2 dataframe with rows (ens, gene symbol)
    blacklist : str
        path to file with ens and gene symbol column for blacklisted genes
    blacklist_key : str
        name of column to use in whitelist

    Returns
    -------
    df_filt : dataframe
        expression dataframe with only accepted genes
    genes_filt : dataframe
        genes remaining after filter
    """
    reject = scio.load_genes(blacklist)
    # ignore '.[NUMBER]' in ensembl gene ids
    reject[blacklist_key] = reject[blacklist_key].str.split('.').str[0]
    #  ignore genes in blacklist
    mask = ~genes[blacklist_key].str.split('.').str[0].isin(
            reject[blacklist_key].values)
    df_filt, genes_filt = df.loc[mask], genes.loc[mask]
    return df_filt, genes_filt


def get_matrix_stats_dict(df):
    """Get a dictionary of stats about an expression matrix
    Parameters
    ----------
    df : pandas dataframe
        Matrix with genes as rows and cells as columns

    Returns
    -------
    exp_stats : dict
        expression statistics about matrix
    """
    exp_stats = {
        'ngenes' : df.shape[0],
        'ncells' : df.shape[1],
        'mean_mol_per_cell' : float(df.sum(axis=0).mean()),
        'mean_mol_per_gene' : float(df.sum(axis=1).mean()),
        'mean_genes_per_cell' : float(df.astype(bool).sum(axis=0).mean()),
        'mean_cells_per_gene' : float(df.astype(bool).sum(axis=1).mean()),
        'var_mol_per_cell' : float(df.sum(axis=0).var()),
        'var_mol_per_gene' : float(df.sum(axis=1).var()),
        'median_mol_per_cell' : df.sum(axis=0).median(),
        'median_mol_per_gene' : df.sum(axis=1).median(),
        'median_genes_per_cell' : df.astype(bool).sum(axis=0).median(),
        'median_cells_per_gene' : df.astype(bool).sum(axis=1).median(),
        'min_mol_per_cell' : int(df.sum(axis=0).min()),
        'min_mol_per_gene' : int(df.sum(axis=1).min()),
        'max_mol_per_cell' : int(df.sum(axis=0).max()),
        'max_mol_per_gene' : int(df.sum(axis=1).max()),
        'sparsity': 100 * (1 - len(np.nonzero(df.values)[0])/(
            df.shape[0]*df.shape[1]))
        }
    return exp_stats


def prep_and_write_matrix(infile, outdir, prefix, min_cells=10, whitelist='',
        whitelist_key='ens', blacklist='', blacklist_key='ens'):
    """ Filter expression matrix, write stats to YAML file
    Parameters
    ----------
    infile : str
        file with expression matrix formatted: ENS_ID\tGENE\tCELL0\tCELL1\t...
    outdir : str
        directory for output
    prefix : str
        name prefix for output file
    min_cells : int, optional
        minimum number of cells that must express a gene for the gene to be included
        in the filtered matrix. Default 10.
    whitelist : str, optional
        file with genes to whitelist.  formated in two whitespace delimitted columns,
        (ens, gene) with no header. Superseded by blacklist.
    whitelist_key : str, optional
        name of column in whitelist file to use for filtering (ens)
    blacklist : str, optional
        file with genes to blacklist.  formated in two whitespace delimitted columns,
        (ens, gene) with no header. Superseeds whitelist (ie genes filtered if in
        blacklist even if also in whitelist).
    blacklist_key : str, optional
        name of column in blacklist file to use for filtering (ens)

    Returns
    -------
    df : dataframe
        filtered expression dataframe
    genes : dataframe
        genes from filtered expression dataframe

    """
    # load the expression matrix
    df, genes = scio.load_exp_matrix(infile)

    # filter the expression matrix
    # remove genes without minimum num cells expressing
    print('...filtering by number of cells with nonzero expression')
    df_filt, genes_filt = filter_min_cells_expressing(df, genes, min_cells)
    # remove genes not on a given whitelist, if specified
    print('...removing genes not on whitelist')
    if(len(args.whitelist)):
        df_filt, genes_filt = filter_whitelist(df_filt, genes_filt,
                whitelist, whitelist_key)
    # remove genes in blacklist, if specified
    print('...removing genes on blacklist')
    if(len(args.blacklist)):
        df_filt, genes_filt = filter_blacklist(df_filt, genes_filt,
                blacklist, blacklist_key)

    # write the matrix
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    print('...writing filtered matrix to file')
    scio.write_exp_matrix(genes=genes_filt, matrix=df_filt,
            outfile='{0}/{1}{2}matrix.txt'.format(outdir, prefix.rstrip('.'),
                '.' if len(prefix)>0 else ''),
            outfile_gene='{0}/{1}{2}genes.txt'.format(outdir, prefix.rstrip('.'),
                '.' if len(prefix)>0 else ''),
            )

    return df_filt, genes_filt,


def convert_and_split_frac(data, validation_frac=0.02, test_frac=0,
        validation_weighted=False, test_weighted=False):
    """ Split dataset into sparse train, validation, and test sets
    Parameters
    ----------
    data : numpy array
        The entire dataset. A gene x cell matrix
    validation_frac : float, optional
        The fraction of entries to put in the validation set
    test_frac : float, optional
        The fraction of entries to put in the test set
    validation_weighted: bool, optional
        Should sampling of validation set be inversely weighted by the number of
        molecules
    test_weighted: bool, optional
        Should sampling of test set be inversely weighted by the number of
        molecules

    Returns
    -------
    train : np.array
        nsamples x 3 np.array
    validation : np.array or None
        nsamples x 3 np.array
    test : np.array or None
        nsamples x 3 np.array
    """

    nz_gene, nz_cell = np.nonzero(data)
    nz_val = data[nz_gene, nz_cell]
    nz_n = nz_val.shape[0]

    num_vld, num_tst = int(validation_frac*nz_n), int(test_frac*nz_n)
    if num_tst + num_vld == 0:
        test = np.array([])
        valid = np.array([])
        train = np.stack([nz_cell, nz_gene, nz_val]).T
    else:
        # get selection weight functions
        def get_invweights(ixs):
            p_unnorm = 1 / nz_val[ixs]
            return p_unnorm / np.sqrt(p_unnorm.sum())
        get_p_vld = lambda i: get_invweights(i) if validation_weighted else None
        get_p_tst = lambda i: get_invweights(i) if test_weighted else None
        get_data = lambda i: np.stack([nz_cell[i], nz_gene[i], nz_val[i]]).T

        # get validation set
        options = np.arange(nz_val.shape[0])
        vld_ix1d = np.random.choice(options, num_vld, replace=False,
                p=get_p_vld(options))
        vld_ix1d.sort()
        valid = get_data(vld_ix1d)

        # get test set (after updating for validation)
        options = np.setdiff1d(options, vld_ix1d, assume_unique=True)
        tst_ix1d = np.random.choice(options, num_tst, replace=False,
                p=get_p_tst(options))
        tst_ix1d.sort()
        test = get_data(tst_ix1d)

        # get train set from remaining
        trn_ix1d = np.setdiff1d(options, tst_ix1d, assume_unique=True)
        trn_ix1d.sort()
        train = get_data(trn_ix1d)

    return train, valid, test


def leave_n_out(train, n=1):
    """Get n nonzero datapoints per cell for test and or validation sets
    Parameters
    ----------
    train : numpy array
        sample x 3 array with entries like [cell_id, gene_id, mol_count]
    n : int, optional

    Returns
    -------
    train : numpy array
    heldout : numpy array or None
    """
    if n == 0:
        return train, None
    else:
        options = np.arange(train.shape[0])
        cell_ids = train[:, 0]
        selected = []
        for i in np.unique(cell_ids):
            selected_i = np.random.choice(options[cell_ids == i], n,
                    replace=False)
            selected.append(selected_i)
        heldout_ix1d = np.hstack(selected)
        train_ix1d = np.setdiff1d(options, heldout_ix1d, assume_unique=True)

        return train[train_ix1d, :], train[heldout_ix1d, :]


def split_dataset_hpf(data, outdir='', validation_frac=0.01, test_frac=0.05,
        validation_weighted=False, test_weighted=False,
        validation_leave_n_out=0, test_leave_n_out=0):
    """ Split dataset into sparse train, validation, and test sets, and write to
        file in output dir as train.tsv, validation.tsv, and test.tsv
    Parameters
    ----------
    data : numpy array
        The entire dataset. A gene x cell array
    outdir : str, optional
        output directory. not written if empty string
    validation_frac : float, optional
        The fraction of entries to put in the validation set
        Must be 0 if `validation_leave_n_out` is nonzero.
    test_frac : float, optional
        The fraction of entries to put in the test set
        Must be 0 if `test_leave_n_out` is nonzero.
    validation_weighted: bool, optional
        Should sampling of validation set be inversely weighted by the number of
        molecules. Only applied to `validation_frac`, ignored for
        `validation_leave_n_out`.
    test_weighted: bool, optional
        Should sampling of test set be inversely weighted by the number of
        molecules. Only applied to `test_frac`, ignored for
        `test_leave_n_out`.
    validation_leave_n_out : int, optional
        Number of entries per cell to put in the validation set.
        Must be 0 if `validation_fraction` is nonzero.
    test_leave_n_out int, optional
        The number of entries per cell to put in the test set
        Must be 0 if `test_fraction` is nonzero.

    Returns
    -------
    info : dict
        dictionary of information about split dataset

    Note
    ----
    Method allows you to use leave-n-out on one split and fractional on another,
    but that is weird and probably inadvisable
    """
    if validation_leave_n_out > 0 and validation_frac > 0:
        msg = 'One of `validation_leave_n_out` and `validation_frac` must be 0,'
        msg += ' but received {} and {}'
        raise ValueError(msg.format(validation_leave_n_out, validation_frac))
    if test_leave_n_out > 0 and test_frac > 0:
        msg = 'One of `test_leave_n_out` and `test_frac` must be 0,'
        msg += ' but received {} and {}'
        raise ValueError(msg.format(test_leave_n_out, test_frac))

    train, valid, test = convert_and_split_frac(data,
            validation_frac=validation_frac, test_frac=test_frac,
            validation_weighted=validation_weighted,
            test_weighted=test_weighted,)

    if validation_leave_n_out > 0:
        train, valid = leave_n_out(train, validation_leave_n_out)
    if test_leave_n_out > 0:
        train, test = leave_n_out(train, test_leave_n_out)

    # get dataset info
    info = {
        'train_samples': train.shape[0],
        'train_molecules': int(train[:,2].sum()),
        'validation_samples': valid.shape[0] if len(valid) else 0,
        'validation_molecules': int(valid[:,2].sum()) if len(valid) else 0,
        'test_samples':  test.shape[0] if len(test) else 0,
        'test_molecules': int(test[:,2].sum()) if len(test) else 0
    }
    train_stats = {
            'train_' + k : v for k,v in get_matrix_stats_dict(
                pd.DataFrame(train).pivot(index=1, columns=0, values=2
                    ).fillna(0)).items()
            }

    # write datasets
    if len(outdir):
        info['train_file'] = outdir + '/train.tsv'
        np.savetxt(info['train_file'], train, delimiter='\t', fmt='%i')
        if len(valid):
            info['validation_file'] = outdir + '/validation.tsv'
            np.savetxt(info['validation_file'], valid, delimiter='\t', fmt='%i')
        if len(test):
            info['test_file'] = outdir + '/test.tsv'
            np.savetxt(info['test_file'], test, delimiter='\t', fmt='%i')
    return {**train_stats, **info}


def _parser(subparsers=None):
    """Make a new parser, or add prep subparser to an existing one
    Parameters
    ----------
    subparsers : optional
        result of parser.add_subparsers(dest='command')
    Returns
    -------
    prep : parser or subparser
        type depends on `subparsers`.  command='prep' if subparser.

    """
    if subparsers is None:
        prep = argparse.ArgumentParser()
        # subparsers = parser.add_subparsers(dest='command')
    else:
        prep = subparsers.add_parser('prep')

    prep.add_argument('-i', '--input', type=str, required=True,
            help='Input expression matrix formatted: ENS_ID\tGENE\tCELL0\tCELL1\t...')
    prep.add_argument('-o', '--outdir', type=str, required=True)
    prep.add_argument('-p', '--prefix', type=str, required=True)
    prep.add_argument('-m', '--min-cells', type=float, default=10, required=False,
            help='Minimum number of cells in which we must observe at least one'
            ' transcript of a gene for the gene to pass filtering.  '
            'If 0 < `min_cells` < 1, sets threshold to be `min_cells` * ncells.')
    prep.add_argument('-t', '--test-fraction', type=float, default=0.0,
            help='Fraction of nonzero values to use for test set.')
    prep.add_argument('-v', '--validation-fraction', type=float, default=0,
            help='Fraction of nonzero values to use for validation set.')
    prep.add_argument('-tn', '--test-leave-n-out', type=int, default=0,
            help='Number of nonzero counts per cell to use in test set.')
    prep.add_argument('-vn', '--validation-leave-n-out', type=int, default=0,
            help='Number of nonzero counts per cell to use in validation set.')
    prep.add_argument('-tw', '--test-weighted', default=False, action='store_true',
            help='Weight probability of test set selection by inverse of value.')
    prep.add_argument('-vw', '--validation-weighted', default=False,
            action='store_true',
            help='Weight probability of validation set selection by inverse of '
                'value.')
    prep.add_argument('-w', '--whitelist', default='',
            help='File where first column is list of ensembl gene ids to accept,'
             ' and second column is gene symbol. Only performed if file '
             'given. Superseded by blacklist')
    prep.add_argument('-b', '--blacklist', default='',
            help='File where first column is list of ensembl gene ids to exclud,'
             ' and second column is gene symbol. Only performed if file '
             'given. Supersedes whitelist.')
    prep.add_argument('--filter-by-gene-name', default=False, action='store_true',
            help='Use gene name rather than ens symbol to filter.  Applies to '
            'both whitelist and blacklist.')
    return prep


def _check_args(args):
    """Check args
    """
    if args.validation_leave_n_out > 0 and args.validation_fraction > 0:
        msg = 'One of `validation_leave_n_out` and `validation_frac` must be 0,'
        msg += ' but received {} and {}'
        raise ValueError(msg.format(args.validation_leave_n_out,
            args.validation_fraction))
    if args.test_leave_n_out > 0 and args.test_fraction > 0:
        msg = 'One of `test_leave_n_out` and `test_frac` must be 0,'
        msg += ' but received {} and {}'
        raise ValueError(msg.format(args.test_leave_n_out, args.test_fraction))

    if args.validation_leave_n_out and args.validation_weighted:
        msg = 'Warning: `validation_weighted` ignored when '
        msg += '`validation_leave_n_out` > 0'
        print(msg)
        args.validation_weighted = False
    if args.test_leave_n_out and args.test_weighted:
        msg = 'Warning: `test_weighted` ignored when `test_leave_n_out` > 0'
        print(msg)
        args.test_weighted = False


if __name__=='__main__':
    parser = _parser()
    args = parser.parse_args()
    _check_args(args)

    df, genes = prep_and_write_matrix(infile=args.input,
            outdir=args.outdir, prefix=args.prefix, min_cells=args.min_cells,
            whitelist=args.whitelist, blacklist=args.blacklist,
            whitelist_key='gene' if args.filter_by_gene_name else 'ens',
            blacklist_key='gene' if args.filter_by_gene_name else 'ens'
            )
    exp_stats = get_matrix_stats_dict(df)
    # make train, test, and validation datasets

    split_stats = split_dataset_hpf(data=df.values, outdir=args.outdir,
            validation_frac=args.validation_fraction,
            validation_leave_n_out=args.validation_leave_n_out,
            validation_weighted=args.validation_weighted,
            test_frac=args.test_fraction, test_leave_n_out=args.test_leave_n_out,
            test_weighted=args.test_weighted)

    # write preprocessing parameters to file
    dargs = dict(list(vars(args).items()) + list(exp_stats.items()) \
                + list(split_stats.items()))
    with open('{}/preprocessing.log.yaml'.format(dargs['outdir']), 'w') as f:
        yaml.dump(dargs, f, default_flow_style=False)
