#!/usr/bin/python

import sys
import os
import gc
import glob
import warnings
import argparse

import yaml
import numpy as np
import pandas as pd
import tensorflow as tf

from .util import get_session, create_sparse_tensor
from .scio import write_exp_matrix, load_sparse_exp
from .hpf_params import HyperParams, VariationalParams


class plots:
    """Container class for plotting"""

    def _import_plotlibs(for_save=False):
        """Annoying trick to work both in ipython notebook and scripts"""
        if for_save:
            import matplotlib as mpl
            mpl.use('agg')
            mpl.rcParams['pdf.fonttype'] = 42
            mpl.rcParams['ps.fonttype'] = 42
            import matplotlib.pyplot as plt
            plt.switch_backend('agg')
            warnings.filterwarnings("ignore", category=UserWarning,
                module="matplotlib")
        else:
            import matplotlib as mpl
            import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style('ticks')
        return mpl, plt, sns


    @staticmethod
    def loading_dist_plot(vp, metric='norm', garbage_collect=False, nbins=100,
            seaborn_context_params={'context':'notebook'}, outdir='', prefix='',
            save_options={'bbox_inches':'tight'}):
        """Create histograms of maximum metric (based on loading) for cells and genes

        Parameters
        ----------
        vp : VariationalParams
            variational distribution for the model
        metric : str, default 'norm'
            Metric to historgram the max value of over each cell (left plot)
            or gene (right plot). Valid values are:
                'norm' : E[loading|data] / E[sum(loading)|data]
                    where sum is over all loadings for a cell or gene
                'hnorm': E[loading|data] * E[capacity|data]
                         (score in scHPF manuscript)
                'geonorm' : E[loading|data] / geometric_mean(E[loading|data])
                    where the geometric mean is over all loadings for a cell gene
                'termscore' : `norm` * log(`geonorm`)
        garbage_collection : bool, default False
            Garbage collect after plotting
        nbins : int, default 100
            number of bins per histogram
        seaborn_context_params : dict
            Dictionary kwargs to pass to sns.set_context
        outdir : str, default ''
            Directory to save image to.  No save if len(`outdir`) == 0
        prefix : str, default ''
            Filename prefix during saving.  Can be left as ''
        save_options : dict
            Dictionary of kwargs to pass to fig.savefig

        Returns
        -------
        fig : matplotlib figure
        (ax0,  ax1) : matplotlib axes for cell and gene histograms, respectively
        """
        sess = get_session()
        if metric == 'norm':
            tm = tf.reduce_max(vp.theta.norm, axis=1)
            bm = tf.reduce_max(vp.beta.norm, axis=1)
            tlab = 'Max cell weight proportion'
            blab = 'Max gene weight proportion'
        elif metric == 'hnorm':
            tm = tf.reduce_max(vp.xinorm, axis=1)
            bm = tf.reduce_max(vp.etanorm, axis=1)
            tlab = 'Max hierarchically normalized cell weight'
            blab = 'Max hierarchically normalized gene weight'
        elif metric == 'termscore':
            tm = tf.reduce_max(vp.theta.termscore, axis=1)
            bm = tf.reduce_max(vp.beta.termscore, axis=1)
            tlab = 'Max cell termscore'
            blab = 'Max gene termscore'
        elif metric == 'geonorm':
            tm = tf.reduce_max(vp.theta.geonorm, axis=1)
            bm = tf.reduce_max(vp.beta.geonorm, axis=1)
            tlab = 'Max geonormalized cell weight'
            blab = 'Max geonormalized gene weight'
        else:
            message = 'Valid values for metric are '
            message += '{`norm`, `hnorm`, `termscore`, `geonorm`}.'
            message +=  ' Received {0}'.format(metric)
            raise InvalidArgumentException(message)

        mpl, plt, sns = plots._import_plotlibs(len(outdir) > 0)
        sns.set_context(**seaborn_context_params)
        fig, (a0,a1) = plt.subplots(1,2)
        a0.hist(sess.run(tm), nbins, linewidth=0, edgecolor='none',
                rasterized=True)
        sns.despine()

        a0.set_title(tlab)
        a1.hist(sess.run(bm), nbins, linewidth=0, edgecolor='none',
                rasterized=True)
        sns.despine()
        a1.set_title(blab)

        if len(outdir):
            outfile = '{0}/{1}{2}.pdf'.format(outdir, prefix, metric)
            fig.tight_layout()
            fig.savefig(outfile, format='pdf', **save_options)

        if garbage_collect:
            del tm, bm
            gc.collect()
        return fig, (a0, a1)


    @staticmethod
    def capacity_correlation_plot_tf(vp, data, outdir='', prefix='',
            seaborn_context_params={'context':'talk'},
            save_options={'rasterized':True, 'bbox_inches':'tight'},
            feed_dict={}):
        """ Plot 1/E[capacity|data] vs molecules per gene or cell correlations

        Parameters
        ----------
        vp : VariationalParams
            variational distribution for the model
        data: tf sparse tensor or sparse placeholder
            The tf sparse tensor or placeholder plot on row and columns sums
            of on x axis
        outdir : str, default ''
            Directory to save image to.  No save if len(`outdir`) == 0
        prefix : str, default ''
            Filename prefix during saving.  Can be left as ''
        seaborn_context_params : dict
            Dictionary kwargs to pass to sns.set_context
        save_options : dict
            Dictionary of kwargs to pass to fig.savefig
        feed_dict : dict
            feed_dict for data

        Returns
        -------
        g_xi: Seaborn jointplot for E[xi|data] (yaxis) vs cell_sum(data)
        g_eta: Seaborn jointplot for E[eta|data] (yaxis) vs gene_sum(data)

        """
        mpl, plt, sns = plots._import_plotlibs(len(outdir) > 0)
        sess = get_session()
        cell_sum, gene_sum = sess.run([
            tf.sparse_reduce_sum(data, axis=1, keep_dims=True),
            tf.transpose(tf.sparse_reduce_sum(data, axis=0, keep_dims=True))],
            feed_dict=feed_dict)
        xi, eta = sess.run([vp.xi.e_x, vp.eta.e_x])
        # adjust for cells and genes not present in data
        xi = xi[cell_sum > 0]
        cell_sum = cell_sum[cell_sum > 0]
        eta = eta[gene_sum > 0]
        gene_sum = gene_sum[gene_sum > 0]

        sns.set_context(**seaborn_context_params)
        g_xi = sns.jointplot(np.log2(cell_sum), np.log2(1/xi),
                joint_kws={'facecolor':'none'})
        g_xi.ax_joint.set_xlabel('log2 molecules per cell')
        g_xi.ax_joint.set_ylabel('log2 inferred cell normalizer')
        g_xi.fig.show()
        if len(outdir):
            outfile = '{0}/{1}{2}.pdf'.format(outdir, prefix, 'cell_correlation')
            fig = g_xi.fig
            fig.tight_layout()
            fig.savefig(outfile, format='pdf', **save_options)

        g_eta = sns.jointplot(np.log2(gene_sum), np.log2(1/eta),
                joint_kws={'facecolor':'none'})
        g_eta.ax_joint.set_xlabel('log2 molecules per gene')
        g_eta.ax_joint.set_ylabel('log2 inferred gene normalizer')
        g_eta.fig.show()
        if len(outdir):
            outfile = '{0}/{1}{2}.pdf'.format(outdir, prefix, 'gene_correlation')
            fig = g_eta.fig
            fig.savefig(outfile, format='pdf', **save_options)

        return g_xi, g_eta


def factor_metrics(vp, data, dom_p=0.8, feed_dict={}):
    tve = vp.total_variation_explained(data)
    fve = vp.factor_variation_explained(data)
    ts2e = vp.total_variance_explained(data)
    dp = vp.dominance_at_p(data, p=dom_p)

    sess = get_session()
    tve_, fve_, ts2e_, dp_ = sess.run([tve, fve, ts2e, dp], feed_dict=feed_dict)
    nonneg_fve_ = bool(np.all(np.equal(fve_, np.abs(fve_))))

    metrics = {'nfactors': vp.hyper_p.nfactors,
               'total_variation_explained' : tve_.tolist(),
               'factor_variation_explained' : fve_.tolist(),
               'total_variance_explained' : ts2e_.tolist(),
               'nonnegative_fve' : nonneg_fve_,
               'dominance_p' : dom_p,
               'dominance_at_p' : dp_.tolist(),
              }
    return metrics


def write_factor_metrics(vp, data, outdir, prefix='', dom_p=0.8, feed_dict={}):
    metrics = factor_metrics(vp, data, dom_p, feed_dict)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = '{0}/{1}factor_metrics.yaml'.format(outdir, prefix)
    with open(outfile, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)


def _parser(subparsers=None):
    if subparsers is None:
        parser  = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')

    metrics = subparsers.add_parser('metrics')
    metrics.add_argument('-p', '--param-dir', required=True)
    metrics.add_argument('-d', '--data-dir', required=True)
    metrics.add_argument('--prefix', default='')
    metrics.add_argument('-o', '--outdir', default=None)
    metrics.add_argument('--dom-p', default=0.8)

    score = subparsers.add_parser('score')
    score.add_argument('-p', '--param-dir', required=True)
    score.add_argument('-o', '--outdir', default='')
    score.add_argument('-m', '--metric', default='hnorm',
        choices=['norm', 'hnorm', 'termscore', 'all'],
        help="Score function to write. Default `hnorm` (hierarchical "
            "normalization), which is the score function used in the scHPF "
            "manuscript."
            )
    score.add_argument('--prefix', default='')
    score.add_argument('-npy', default=False, action='store_true',
        help="Store values in npy files instead of whitespace delimited txt files")
    score.add_argument('--save-exp', action='store_true', default=False,
            help='Save expectations (the posterior means) in a subdirectory.')

    img = subparsers.add_parser('img')
    img.add_argument('-p', '--param-dir', required=True)
    img.add_argument('-d', '--data-dir', required=True)
    img.add_argument('--prefix', default='')
    img.add_argument('-o', '--outdir', default=None)
    return parser


if __name__=='__main__':
    parser = _parser()
    args = parser.parse_args()

    if args.command=='metrics':
        if args.outdir is None:
            args.outdir = args.param_dir

        print('loading parameters...')
        vp = VariationalParams.load_from_file(args.param_dir,
                prefix=args.prefix)

        print('...loading data')
        train_file = os.path.join(args.data_dir,'train.tsv')
        ntrn, ncells, ngenes, indices, values = load_sparse_exp(train_file)
        shape=np.array([ncells, ngenes])
        data = tf.sparse_placeholder(dtype=vp.hyper_p.dtype, name='trn_data')
        data_dict = {data : tf.SparseTensorValue(indices=indices, values=values,
            dense_shape=shape)}

        validation_file = os.path.join(args.data_dir,'validation.tsv')
        if os.path.exists(validation_file):
            nvld, _, _, vindices, vvalues = load_sparse_exp(validation_file)
            vdata = tf.sparse_placeholder(dtype=vp.hyper_p.dtype, name='vld_data')
            data_dict[vdata] = tf.SparseTensorValue(indices=vindices,
                    values=vvalues, dense_shape=shape)
        else:
            nvld, vdata = 0, None

        test_file = os.path.join(args.data_dir,'test.tsv')
        if os.path.exists(test_file):
            ntest, _, _, tindices, tvalues = load_sparse_exp(test_file)
            tdata = tf.sparse_placeholder(dtype=vp.hyper_p.dtype, name='tst_data')
            data_dict[tdata] = tf.SparseTensorValue(indices=tindices,
                    values=tvalues, dense_shape=shape)
        else:
            ntst, tdata = 0, None

        with tf.name_scope('alldata'):
            if vdata is not None and tdata is None:
                alldata  = tf.sparse_add(data, vdata)
            elif vdata is None and tdata is not None:
                alldata  = tf.sparse_add(data, tdata)
            elif vdata is not None and tdata is not None:
                alldata  = tf.sparse_add(tf.sparse_add(data, tdata), vdata)
            else:
                alldata = data

        all_data_prefix = '{0}all.'.format(args.prefix)
        trn_data_prefix = '{0}trn.'.format(args.prefix)
        tst_data_prefix = '{0}tst.'.format(args.prefix)
        vld_data_prefix = '{0}vld.'.format(args.prefix)

        sess = get_session()
        init = tf.global_variables_initializer()
        sess.run(init)

        print('...computing factor metrics')
        write_factor_metrics(vp, data=alldata, outdir=args.outdir,
                prefix=all_data_prefix, dom_p=args.dom_p,
                feed_dict=data_dict)
        if vdata is not None or tdata is not None:
            write_factor_metrics(vp, data=data, outdir=args.outdir,
                    prefix=trn_data_prefix, dom_p=dom_p,
                    feed_dict=data_dict)
        if vdata is not None:
            write_factor_metrics(vp, data=vdata, outdir=args.outdir,
                    prefix=vld_data_prefix, dom_p=dom_p,
                    feed_dict=data_dict)
        if tdata is not None:
            write_factor_metrics(vp, data=tdata, outdir=args.outdir,
                    prefix=tst_data_prefix, dom_p=dom_p,
                    feed_dict=data_dict)

    elif args.command in ['score', 'img']:
        print('loading parameters...')
        vp = VariationalParams.load_from_file(args.param_dir,
                prefix=args.prefix)

        sess = get_session()
        init = tf.global_variables_initializer()
        sess.run(init)

        if args.command == 'score':
            if args.outdir is None or args.outdir == '':
                args.outdir = args.param_dir + '/score'
            if args.save_exp:
                vp.write_params_to_file(args.outdir + '/expectations',
                        npy=args.npy, save_variational=False,
                        save_expectations=True)
            vp.write_score_to_file(outdir=args.outdir, prefix=args.prefix,
                    score=args.metric, npy=args.npy)

        elif args.command == 'img' :
            if args.outdir is not None:
                imgdir = args.outdir
            else:
                basename = args.param_dir.rstrip('/').rsplit('/', 1)[1]
                imgdir = '{0}/{1}.img'.format(args.param_dir, basename)
            os.makedirs(imgdir, exist_ok=False)
            nbins = min(100, int(vp.hyper_p.ncells/20))

            plots.loading_dist_plot(vp, metric='norm', outdir=imgdir,
                    prefix=args.prefix, nbins=nbins, garbage_collect=True)
            plots.loading_dist_plot(vp, metric='hnorm', outdir=imgdir,
                    prefix=args.prefix, nbins=nbins, garbage_collect=True)

            # load data
            train_file = os.path.join(args.data_dir,'train.tsv')
            nsamples, ncells, ngenes, indices, values = load_sparse_exp(
                    train_file)
            trn_data = create_sparse_tensor(indices=indices, values=values,
                    ncells=ncells, ngenes=ngenes, dtype=vp.hyper_p.dtype)
            validation_file = os.path.join(args.data_dir,'validation.tsv')
            if os.path.exists(validation_file):
                nvld, _, _, vindices, vvalues = load_sparse_exp(validation_file)
                vld_data = create_sparse_tensor(indices=vindices,
                        values=vvalues, ncells=ncells, ngenes=ngenes,
                        dtype=vp.hyper_p.dtype)
            else:
                nvld, vld_data = 0, None
            test_file = os.path.join(args.data_dir,'test.tsv')
            if os.path.exists(test_file):
                ntst, _, _, tindices, tvalues = load_sparse_exp(test_file)
                tst_data = create_sparse_tensor(indices=tindices,
                        values=tvalues, ncells=ncells, ngenes=ngenes,
                        dtype=vp.hyper_p.dtype)
            else:
                ntst, tst_data = 0, None
            if tst_data is None and vld_data is None:
                all_data = trn_data
            elif tst_data is None and vld_data is not None:
                all_data = tf.sparse_add(trn_data, vld_data)
            elif tst_data is not None and vld_data is None:
                all_data = tf.sparse_add(trn_data, tst_data)
            else:
                all_data = tf.sparse_add(tst_data,
                        tf.sparse_add(trn_data, vld_data))

            # make capacity correlation plots
            all_data_prefix = '{0}.all'.format(args.prefix.rstrip('.'))
            trn_data_prefix = '{0}.trn'.format(args.prefix.rstrip('.'))
            plots.capacity_correlation_plot_tf(vp, data=all_data, outdir=imgdir,
                    prefix=all_data_prefix)
            plots.capacity_correlation_plot_tf(vp, data=trn_data, outdir=imgdir,
                    prefix=trn_data_prefix)
            if tst_data is not None:
                tst_data_prefix = '{0}.tst'.format(args.prefix.rstrip('.'))
                plots.capacity_correlation_plot_tf(vp, data=tst_data,
                        outdir=imgdir, prefix=tst_data_prefix)
            if vld_data is not None:
                vld_data_prefix = '{0}.vld'.format(args.prefix.rstrip('.'))
                plots.capacity_correlation_plot_tf(vp, data=vld_data,
                        outdir=imgdir, prefix=vld_data_prefix)


