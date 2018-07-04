#!/usr/bin/python

import os
import glob
import shutil
import json
import argparse
from collections import defaultdict

import yaml
import numpy as np
import tensorflow as tf

from scio import load_sparse_exp
from util import create_sparse_tensor
from hpf_params import HyperParams, VariationalParams
from hpf_inference import HPFInference
from postprocessing import factor_metrics, plots, write_factor_metrics


def run_trials(indir, outdir, prefix, nfactors, a, ap, bp, c, cp, dp, ntrials=1,
        max_iter=1000, min_iter=30, epsilon=0.01, rfreq=10, logging_options={},
        dtype=tf.float64, loss_name='llh', write_all=False, dom_p=0.8,
        better_than_n_ago=5, save_all_logs=False, save_img=False,
        bestm_name='loss', phi_init_param=[], save_init=False,
        save_inflection=False, compute_factor_metrics=True):
    """ Run scHPF training multiple times, selecting the run which results in the
        greatest ELBO
    Parameters
    ----------
    indir : str
        Input directory with train.tsv, a sparse coo matrix of expression values.
    outdir : str
        output directory
    prefix : str
        prefix for output files
    nfactors : int
        Number of factors
    a : float
        shape prior for cell factor loadings
    ap : float
        shape prior for cell capacities
    bp : float
        inverse rate prior for cell capacities
    c : float
        shape prior for gene factor loadings
    cp : float
        shape prior for gene capacity loadings
    dp : float
        inverse rate prior for gene capacities
    phi_init_param : np array, optional
        numpy array with concentration parameters to initialize normalized
        auxiliary parameters.  Must have len 0 or len == nfactors.
    ntrials : int , optional
        Number of trials
    max_iter : int , optional
        maximum number of iterations
    min_iter : int , optional
        minimum iterations
    epsilon : float , optional
        percent change of loss for convergence
    rfreq : int , optional
        iterations between checking convergence
    better_than_n_ago : int, default 5
        stop if loss worse than n*freq steps ago and getting worse
    logging_options : dict, optional
        logging options for HPFInference
    rm_rejected : bool , optional
        remove trials with suboptimal elbos
    dtype : tensorflow dtype, optional
        dtype of latent variables. Default tf.float64
    loss_name: str, optional
        Stat to use to assess convergence.  Valid values are: {'elbo','llh'}
    dom_p : float, optional
        Proportion of explained expression at which to assess dominace, range
        (0,1).  Stats are not computed if None.  [default 0.8]
    save_all_logs : bool, optional
        save run info (and images) even if not write_all
    save_img : bool, optional
        save figures for run
    save_init : bool [false]
        save the model initialization
    save_inflection : bool [false]
        save the model at inflection points
    bestm_name : str, optional
        metric to select best run

    Returns
    -------
    best_run_dir : str
        directory of best run
    """
    # load data and setup placeholders
    train_file = os.path.join(indir,'train.tsv')
    ntrn, ncells, ngenes, indices, values = load_sparse_exp(train_file)
    print('indices shape', indices.shape)
    shape=np.array([ncells, ngenes])
    data = tf.sparse_placeholder(dtype=dtype, name='trn_data')
    data_dict = {data : tf.SparseTensorValue(indices=indices, values=values,
        dense_shape=shape)}

    validation_file = os.path.join(indir,'validation.tsv')
    if os.path.exists(validation_file):
        nvld, _, _, vindices, vvalues = load_sparse_exp(validation_file)
        vdata = tf.sparse_placeholder(dtype=dtype, name='vld_data')
        data_dict[vdata] = tf.SparseTensorValue(indices=vindices, values=vvalues,
                dense_shape=shape)
    else:
        nvld, vdata = 0, None

    test_file = os.path.join(indir,'test.tsv')
    if os.path.exists(test_file):
        ntest, _, _, tindices, tvalues = load_sparse_exp(test_file)
        tdata = tf.sparse_placeholder(dtype=dtype, name='tst_data')
        data_dict[tdata] = tf.SparseTensorValue(indices=tindices, values=tvalues,
                dense_shape=shape)
    else:
        ntst, tdata = 0, None

    # i wish sparse assign add were a thing
    with tf.name_scope('alldata'):
        if vdata is not None and tdata is None:
            alldata  = tf.sparse_add(data, vdata)
        elif vdata is None and tdata is not None:
            alldata  = tf.sparse_add(data, tdata)
        elif vdata is not None and tdata is not None:
            alldata  = tf.sparse_add(tf.sparse_add(data, tdata), vdata)
        else:
            alldata = data

    # create hyperparameter and parameter objects
    with tf.variable_scope('model'):
        hp = HyperParams(nfactors=nfactors, ncells=ncells, ngenes=ngenes, a=a,
                ap=ap, bp=bp, c=c, cp=cp, dp=dp, dtype=dtype)
        vp_kwargs = dict(hyper_p=hp, nsamples=ntrn)
        if phi_init_param is not None and len(phi_init_param) > 0:
            assert(len(phi_init_param) == hp.nfactors)
            vp_kwargs['phi_prior_init'] = phi_init_param
        print('...creating variational distribution')
        vp = VariationalParams.init_random(**vp_kwargs)

    # create inference object
    inference = HPFInference(trn_data=data, vld_data=vdata, tst_data=tdata,
            vi_prm=vp, hyper_prm=hp, logging_options=logging_options,
            loss=loss_name)

    # setup objective functions for convergence
    # a little convoluted... fix at some point
    obj = defaultdict(list)
    if bestm_name not in ['loss', 'llh', 'elbo', 'tve', 'ts2e', 'mae']:
        print('Bad bestm_name {0}.  Using `loss`'.format(bestm_name))
        bestm_name = 'loss'
    bestm_name = loss_name if bestm_name == 'loss' else bestm_name
    if vdata is None:
        print('... no validation data found, using training data to track'
              + ' performance')
        conv_stats = inference.trn_stats
        conv_dict = {data : data_dict[data]}
    else:
        conv_stats = inference.vld_stats
        conv_dict = {vdata : data_dict[vdata]}
    elbo = conv_stats.elbo
    llh = conv_stats.llh_mean
    varexp = conv_stats.varexp
    s2exp = conv_stats.s2exp
    mae = conv_stats.mae

    # setup outer directories and prefixes
    paramdirs, logdirs = [], []
    prefix = prefix.rstrip('.') +'.' if len(prefix)>0 else ''
    my_prefix = '{0}k{1:0>3d}.'.format(prefix, nfactors)
    out_prefix = os.path.join(outdir, my_prefix)

    # check if dir exists
    if len(glob.glob(out_prefix + '*')):
        print('{0}* exists.  Exiting'.format(out_prefix))
        return

    # run trials
    for i in range(ntrials):
        print('Trial {0}...'.format(i))

        # setup output dirs/files
        logdir_i = '{0}/{1}log/{1}run{2}'.format(outdir, my_prefix, i)
        paramdir_i = '{0}/{1}run{2}'.format(outdir, my_prefix, i)
        imgdir = '{0}/{1}run{2}.img'.format(paramdir_i, my_prefix, i)
        runinfo_file = '{0}/{1}run_info.yaml'.format(paramdir_i, my_prefix, i)
        runinfo_file_log = '{0}/{1}run_info.yaml'.format(logdir_i, my_prefix, i)
        try:
            os.makedirs(paramdir_i, exist_ok=False)
            os.makedirs(logdir_i, exist_ok=False)
        except OSError as e:
            problemdir = paramdir_i if os.path.exists(paramdir_i) else logdir_i
            print('{0} exists. Exiting.'.format(problemdir))
            return
        paramdirs.append(paramdir_i)
        logdirs.append(logdir_i)

        # new session
        with tf.Session() as sess:
            # reinitialize inference instance and run
            loss_i = inference.run( trn_tensorvalue=data_dict[data],
                    vld_tensorvalue=data_dict[vdata] if vdata is not None else None,
                    tst_tensorvalue=data_dict[tdata] if tdata is not None else None,
                    rfreq=rfreq, max_iter=max_iter,
                    min_iter=min_iter, epsilon=epsilon, logdir=logdir_i,
                    better_than_n_ago=better_than_n_ago, save_init=save_init,
                    save_inflection=save_inflection)
            obj[loss_name].append(loss_i[-1])

            # get additional objectives for deciding between runs
            obj['tve'].append(sess.run(varexp, feed_dict=conv_dict))
            obj['ts2e'].append(sess.run(s2exp, feed_dict=conv_dict))
            obj['mae'].append(sess.run(mae, feed_dict=conv_dict))
            if bestm_name != loss_name:
                if bestm_name == 'elbo':
                    obj['elbo'] = sess.run(elbo, feed_dict=conv_dict)
                elif bestm_name == 'llh':
                    obj['llh'] = sess.run(llh, feed_dict=conv_dict)

            # get and write basic run info and objectives
            hp_info = hp.to_dict()
            inference_info = {'rfreq' : rfreq,
                              'epsilon' : epsilon,
                              'min_iter' : min_iter,
                              'max_iter' : max_iter,
                              'niter' : int(sess.run(inference.t)),
                              'trial' : i,
                              'loss_type' : inference.loss_name,
                              'loss_final' : float(loss_i[-1]),
                              'indir' : indir,
                              'outdir' : outdir,
                              'logdir' : logdir_i
                             }
            run_info = {**hp_info, **inference_info}
            if tdata is not None:
                run_info['loss_test'] = float(sess.run(inference.loss_tst,
                    feed_dict=data_dict))
            if vdata is not None:
                run_info['loss_train'] = float(sess.run(inference.loss_trn,
                    feed_dict=data_dict))
            with open(runinfo_file, 'w') as f:
                yaml.dump(run_info, f, default_flow_style=False)
            with open(runinfo_file_log, 'w') as f:
                yaml.dump(run_info, f, default_flow_style=False)

            # get best run
            print(bestm_name, obj[bestm_name])
            best = np.argmin(obj[bestm_name]) if bestm_name=='mae' else np.argmax(obj[bestm_name])
            if i == best:
                print('New best [trial {0}] loss: {1}  tve: {2}  ts2e: {3}'.format(
                    i, obj[loss_name][i], obj['tve'][i], obj['ts2e'][i]))
            else:
                print('[trial {0}] loss: {1}  tve: {2}  ts2e: {3}'.format(
                    i, obj[loss_name][i], obj['tve'][i], obj['ts2e'][i]))
                print('Best [trial {0}] loss: {1}  tve: {2}  ts2e: {3}'.format(
                    best, obj[loss_name][best], obj['tve'][best], obj['ts2e'][best]))

            # write model (if best or write_all)
            if i==best or write_all:
                print('...writing model to file')
                vp.write_params_to_file(paramdir_i)

                all_data_prefix = '{0}all.'.format(prefix)
                trn_data_prefix = '{0}trn.'.format(prefix)
                tst_data_prefix = '{0}tst.'.format(prefix)
                vld_data_prefix = '{0}vld.'.format(prefix)

                # write factor metrics
                if dom_p > 0 and compute_factor_metrics:
                    print('...computing factor metrics')
                    write_factor_metrics(vp, data=alldata, outdir=paramdir_i,
                            prefix=all_data_prefix, dom_p=dom_p,
                            feed_dict=data_dict)
                    write_factor_metrics(vp, data=inference.trn_data,
                            outdir=paramdir_i, prefix=trn_data_prefix,
                            dom_p=dom_p, feed_dict=data_dict)
                    if inference.vld_data is not None:
                        write_factor_metrics(vp, data=inference.vld_data,
                                outdir=paramdir_i, prefix=vld_data_prefix,
                                dom_p=dom_p, feed_dict=data_dict)
                    if inference.tst_data is not None:
                        write_factor_metrics(vp, data=inference.tst_data,
                                outdir=paramdir_i, prefix=tst_data_prefix,
                                dom_p=dom_p, feed_dict=data_dict)

                # save imgs
                if save_img:
                    print('...generating images')
                    os.makedirs(imgdir, exist_ok=False)
                    nbins = min(100, int(vp.hyper_p.ncells/20))
                    print('......loading dist plots')
                    plots.loading_dist_plot(vp, metric='norm', outdir=imgdir,
                            prefix=prefix, nbins=nbins, garbage_collect=True)
                    plots.loading_dist_plot(vp, metric='hnorm', outdir=imgdir,
                            prefix=prefix, nbins=nbins, garbage_collect=True)

                    print('......capacity correlation')
                    plots.capacity_correlation_plot_tf(vp, data=inference.trn_data,
                            outdir=imgdir, prefix=trn_data_prefix,
                            feed_dict=data_dict)
                    if inference.tst_data is not None or inference.vld_data is not None:
                        plots.capacity_correlation_plot_tf(vp, data=alldata,
                                outdir=imgdir, prefix=all_data_prefix,
                                feed_dict=data_dict)
                    if inference.tst_data is not None:
                        plots.capacity_correlation_plot_tf(vp,
                                data=inference.tst_data, outdir=imgdir,
                                prefix=tst_data_prefix, feed_dict=data_dict)
                    if inference.vld_data is not None:
                        plots.capacity_correlation_plot_tf(vp,
                                data=inference.vld_data, outdir=imgdir,
                                prefix=vld_data_prefix, feed_dict=data_dict)

    if not write_all:
        print('Removing suboptimal trials...')
        for i in range(len(obj[loss_name])):
            if i != best:
                if not save_all_logs and os.path.exists(logdirs[i]):
                    shutil.rmtree(logdirs[i], ignore_errors=True)
                if os.path.exists(paramdirs[i]):
                    shutil.rmtree(paramdirs[i], ignore_errors=True)
    else:
        #write best to file
        with open(out_prefix + 'best.txt', 'w') as f:
            f.write(str(best))

    return paramdirs[best]


def _parser(subparsers=None):
    """Make a new parser, or add training subparser to an existing one
    Parameters
    ----------
    subparsers : optional
        result of parser.add_subparsers(dest='command')
    Returns
    -------
    hpf : parser or subparser
        type depends on `subparsers`.

    """
    if subparsers is None:
        train = argparse.ArgumentParser()
        # subparsers = parser.add_subparsers(dest='command')
    else:
        train = subparsers.add_parser('train')

    # io parameters
    train.add_argument('-i', '--indir', type=str, required=True, help=
            "Directory with train.tsv")
    train.add_argument('-o', '--outdir', type=str, required=True, help=
            "Directory to write results.")
    train.add_argument('-p', '--prefix', type=str, default='', help=
            "Prefix for output files")

    # model hyperparameters
    train.add_argument('-k', '--nfactors', type=int, required=True, help=
            "Number of hidden factors.")
    train.add_argument('-pre', '--preprocess-log', default='',
            help="yaml preprocessing.log file produced by preprocessing command"
            "command.  Overrides  -bp and -dp with: "
            "     bp = round('train_mean_mol_per_cell' * ap / "
            "       'train_var_mol_per_cell', 10) and "
            "    dp = round('train_mean_mol_per_gene' * cp / "
            "       'train_var_mol_per_gene', 10).")
    train.add_argument('-a', default=0.3, type=float, help='Shape parameter for'
                ' cell factors. -1 => 1/k, -2 => 1/sqrt(k), -3 => 1/ln(k).')
    train.add_argument('-ap', default=1.0, type=float, help='Shape param for cell'
            'capacity. (default 1)')
    train.add_argument('-bp', type=float, help='Cell capacity inverse scale'
            'hyperparam bp. Overridden by -pre option.')
    train.add_argument('-c', default=0.3, type=float, help='Shape parameter for '
        'gene factors. -1 => 1/k, -2 => 1/sqrt(k), -3 => 1/ln(k). (default 0.3)')
    train.add_argument('-cp',  default=1.0, type=float, help='Shape param for gene'
            'capacity. (default 1)')
    train.add_argument('-dp', type=float, help='Gene capacity inverse scale'
            'hyperparam dp. Overridden by -pre option.')

    # training parameters
    train.add_argument('-t', '--ntrials',  type=int, default=1, help='Number of'
        ' times to run scHPF, selecting the trial with best loss on the '
        'validation set, if given, or the best loss on the training set if no'
        ' validation set is given.')
    train.add_argument('-M', '--max-iter', type=int, default=1000, help='Maximum'
        'iterations. Default 1000.')
    train.add_argument('-m', '--min-iter', type=int, default=30, help='Minimum'
        'iterations. Default 30')
    train.add_argument('-e',  '--epsilon', type=float, default=0.001,
        help='Minimum percent decrease in loss between checks to continue '
            'inference (convergence criteria). [Default 0.001].')
    train.add_argument('-r', '--rfreq', type=int, default=10, help='Number of'
        'iterations to run before checking for convergence (since last '
        'check). Default 10.')
    train.add_argument('-l', '--loss', type=str, default='llh',
        choices=['llh', 'mse', 'mae', 'elbo', 'llhw'],
        help='Loss to use to assess convergence.')
    train.add_argument('--bestm', type=str, default='loss',
        choices=['loss', 'tve', 'ts2e', 'mae'],
        help='Loss for model selection.  `loss` results in same function' \
                + ' as given with `-l`/`--loss`')
    train.add_argument('--dtype', default='float64')
    train.add_argument('--clip', dest='clip', default=True, action='store_true',
        help='Clip hyperparameters for numerical stability if they fall outside '
            'a reasonable range.  Default True.')
    train.add_argument('--no-clip', dest='clip', action='store_false',
        help='Don\'t clip  hyperparameters for stability. Not advisable, '
            ' but rarely needed in data with UMIs.')
    train.add_argument('-bna', '--better-than-n-ago', default=5, type=int,
        help='Ignored if <1. Number of epochs ago loss must be better than for'
            ' convergence.  In difficult datasets, this helps keep scHPF from '
            'stopping at bad local minima.')

    # saving
    train.add_argument('--write-all', default=False, action='store_true',
        help='Save all trials. Can cause a lot of files to be written')
    train.add_argument('--save-all-logs', default=False, action='store_true',
        help='Save logs for all trials. Can cause a lot of files to be written.')
    train.add_argument('--save-img', default=False, action='store_true',
        help='Make diagnostic capacity correlation and factor distribution '
            'plots. Useful for assessing model fit.')
    train.add_argument('--save-init', default=False, action='store_true',
        help='Save parameter initialization. For debugging.')
    train.add_argument('--save-inflection', default=False, action='store_true',
        help='Save model at loss inflection points. Useful for debugging, '
             'but can cause many files to be written.')
    train.add_argument('-wfm', '--write-factor-metrics',
        dest='compute_factor_metrics', default=True, type=bool, nargs='?',
        const=True, help='Compute model fit summary statistics.  Takes more '
            'memory than running scHPF.')
    train.add_argument('--low-mem', dest='compute_factor_metrics',
            action='store_false', help='Reduce memory usage by not computing'
                ' summary statistics for model.')

    # logging
    train.add_argument('--log-llh', dest='log_llh', action='store_true',
            help='Log the mean log likelihood', default=True)
    train.add_argument('--no-log-llh', dest='log_llh', action='store_false',
            help='Don\'t log the log likelihood')
    train.add_argument('--log-llhp', dest='log_llhp', action='store_true',
            help='Log the pointwise log likelihood', default=False)
    train.add_argument('--log-elbo', action='store_true',
            help='Log the ELBO')
    train.add_argument('--log-elbo-components', action='store_true',
            help='Log the ELBO components (terms for each random variable)')
    train.add_argument('--log-mse', action='store_true',
            help='log the mean squared error')
    train.add_argument('--log-mae', action='store_true',
            help='log the mean absolute error')
    train.add_argument('--log-xi', action='store_true',
            help='Log xi (histogram)')
    train.add_argument('--log-eta', action='store_true',
            help='Log eta (histogram)')
    train.add_argument('--log-theta', action='store_true',
            help='Log theta (histogram)')
    train.add_argument('--log-beta', action='store_true',
            help='Log beta (histogram)')
    train.add_argument('--log-phi', action='store_true',
            help='Log phi (histogram)')

    return train


def _parseargs_post(args):
    """ Postprocessing on training parser args
    Parameters
    ----------
    args :
        result of parser.parse_args
    Returns
    -------
    args :
        updated arguments preprocessing file, performs clipping, etc.
    """
    if args.preprocess_log is not None and len(args.preprocess_log)==0:
        potential_path = args.indir + '/preprocessing.log.yaml'
        if os.path.exists(potential_path):
            args.preprocess_log = potential_path
    if len(args.preprocess_log):
        if not os.path.exists(args.preprocess_log):
            args.preprocess_log = args.preprocess_log + '/preprocessing.log.yaml'
        try:
            with open(args.preprocess_log, 'r') as stream:
                plog = yaml.load(stream)
                args.bp = round(plog['train_mean_mol_per_cell'] * args.ap/ (
                    plog['train_var_mol_per_cell']), 10)
                args.dp = round(plog['train_mean_mol_per_gene'] * args.cp/ (
                    plog['train_var_mol_per_gene']), 10)

                # clip vales of dp and cp to be within a multiple of max_dif of
                # one another if they are not and args.clip is true. note this
                # condition did not occur in datasets for the scHPF manuscript.
                max_dif = 1000
                if args.clip and args.dp > max_dif * args.bp:
                    args.bp = args.dp / max_dif
                if args.clip and args.bp > max_dif * args.dp:
                    args.dp = args.bp / max_dif
        except OSError as e:
            print(e)
            print('Error loading log file. Attempting to use given params.')
        except yaml.YAMLError as e:
            print(e)
            print('Error reading yaml log file. Attempting to use given params.')
    if args.a == -1 :
        args.a = round(1 / args.nfactors, 4)
    elif args.a == -2 :
        args.a = round(1 / np.sqrt(args.nfactors),4)
    elif args.a == -3 :
        args.a = min(round(1 / np.log(args.nfactors),4), 1.0)
    elif args.a == -4 :
        args.a = min(round(1 / np.log2(args.nfactors),4), 1.0)

    if args.c == -1:
        args.c = round(1 / args.nfactors, 4)
    elif args.c == -2 :
        args.c = round(1 / np.sqrt(args.nfactors),4)
    elif args.c == -3 :
        args.c = min(round(1 / np.log(args.nfactors),4), 1.0)
    elif args.c == -4 :
        args.c = min(round(1 / np.log2(args.nfactors),4), 1.0)

    # clip for numerical stability in downstream calculations
    # not needed for datasets used in scHPF manuscript
    if args.clip:
        args.bp = np.clip(args.bp, 1e-7, 10000)
        args.dp = np.clip(args.dp, 1e-7, 10000)

    if args.dtype == 'float64':
        args.dtype = tf.float64
    elif args.dtype == 'float32':
        args.dtype = tf.float32

    if args.save_img:
        try:
            import seaborn as sns
        except ImportError:
            msg = 'To save images during training, the python package seaborn'
            msg += ' must be installed.  Please install seaborn or remove '
            msg += 'the \'-save-img\' flag.'
            raise ImportError(msg)
    return args


if __name__=='__main__':
    parser = _parser()
    args = parser.parse_args()
    args = _parseargs_post(args)

    param_log = { 'a' : float(args.a),
                  'ap' : args.ap,
                  'bp' : float(args.bp),
                  'c' : float(args.c),
                  'cp' : args.cp,
                  'dp' : float(args.dp),
                  'nfactors' : args.nfactors,
                  'rfreq' : args.rfreq,
                  'min_iter' : args.min_iter,
                  'max_iter' : args.max_iter,
                  'epsilon' : args.epsilon}
    print(json.dumps(param_log))

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    logging_options = dict( log_elbo=args.log_elbo, log_llh=args.log_llh,
            log_mse=args.log_mse, log_mae=args.log_mae,
            log_xi=args.log_xi, log_eta=args.log_eta, log_theta=args.log_theta,
            log_beta=args.log_beta, log_phi=args.log_phi,
            log_elbo_components=args.log_elbo_components,
            log_llhp=args.log_llhp
            )
    run_trials(indir=args.indir, outdir=args.outdir, prefix=args.prefix,
            nfactors=args.nfactors, a=args.a, ap=args.ap, bp=args.bp, c=args.c,
            cp=args.cp, dp=args.dp, ntrials=args.ntrials,
            max_iter=args.max_iter, min_iter=args.min_iter, rfreq=args.rfreq,
            epsilon=args.epsilon, dtype=args.dtype,
            logging_options=logging_options, loss_name=args.loss,
            write_all=args.write_all, save_all_logs=args.save_all_logs,
            save_img=args.save_img, bestm_name=args.bestm,
            save_init=args.save_init, save_inflection=args.save_inflection,
            better_than_n_ago=args.better_than_n_ago,
            compute_factor_metrics=args.compute_factor_metrics,)
