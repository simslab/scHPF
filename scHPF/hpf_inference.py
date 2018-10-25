#!/usr/bin/python

import numpy as np
import tensorflow as tf
import json

from .util import get_session, get_median_1d
from .hpf_metrics import HPFMetrics


class HPFInference(object):
    """ Base class for inference.

    Notes
    -----
    Many aspects of this class were modeled after parts of Dustin Tran's Edward:
    github.com/blei-lab/edward
    """
    def __init__(self, vi_prm, hyper_prm, trn_data, vld_data=None, tst_data=None,
            t=0, loss='elbo', logging_options={}):
        """
        Parameters
        ----------
        vi_prm: VariationalParams
            Variational inference parameters
        hyper_prm: HyperParams
            Hyperparameters for model
        trn_data : SparseTensor
            training data.  kind of gross to include this here, but unfotunately
            tensorflow does not have placeholders for sparse tensors at the moment.
        vld_data : SparseTensor, optional
            validation_data.
        tst_data : SparseTensor, optional
            tst_data.
        t: int, optional
            Current iteration. Default 0, but can be set for reloading old model
        loss: str, optional
            Stat to use to assess convergence.  Valid values are:
            {'elbo','llh', 'llhw'}
        logging_options : dictionary, optional
            options for what to log
        """
        self.t = tf.Variable(t, trainable=False, name='t')

        self.vi_prm = vi_prm
        self.hyper_prm = hyper_prm
        self.loss_name = loss

        self.trn_data = trn_data
        self.vld_data = vld_data
        self.tst_data = tst_data

        # increment iteration counter
        self.increment_t = self.t.assign_add(1)

        # add cavi training updates in order
        with tf.name_scope('inference'):
            self.train = [self._build_capacity_updates()]
            self.train.extend(self._build_loading_updates(self.trn_data))
            self.train.append(self._build_z_updates(self.trn_data.indices))

        # quantities for evaluating model
        # note most of these are only calculated if specifically flagged
        self.trn_stats = HPFMetrics(self.vi_prm, self.trn_data, data_name='trn')
        self.vld_stats = HPFMetrics(self.vi_prm, self.vld_data, data_name='vld',
                elbo_calc_phi=True) if self.vld_data is not None else None
        self.tst_stats = HPFMetrics(self.vi_prm, self.tst_data, data_name='tst',
                elbo_calc_phi=True) if self.tst_data is not None else None

        # TODO put in loop
        # setup convergence statistic and loss
        if 'elbo' == str.lower(self.loss_name):
            self.loss_trn = self.trn_stats.elbo
            if self.vld_data is not None:
                self.loss_vld = self.vld_stats.elbo
            if self.tst_data is not None:
                self.loss_tst = self.tst_stats.elbo
        elif 'llh' == str.lower(self.loss_name):
            self.loss_trn = self.trn_stats.llh_mean
            if self.vld_data is not None:
                self.loss_vld = self.vld_stats.llh_mean
            if self.tst_data is not None:
                self.loss_tst = self.tst_stats.llh_mean
        elif 'llhm' == str.lower(self.loss_name):
            self.loss_trn = self.trn_stats.llh_median
            if self.vld_data is not None:
                self.loss_vld = self.vld_stats.llh_median
            if self.tst_data is not None:
                self.loss_tst = self.tst_stats.llh_median
        elif 'llhw' == str.lower(self.loss_name):
            self.loss_trn = self.trn_stats.llhw
            if self.vld_data is not None:
                self.loss_vld = self.vld_stats.llhw
            if self.tst_data is not None:
                self.loss_tst = self.llhw_tst
        elif 'mse' == str.lower(self.loss_name):
            self.loss_trn = self.trn_stats.mse
            if self.vld_data is not None:
                self.loss_vld = self.vld_stats.mse
            if self.tst_data is not None:
                self.loss_tst = self.tst_stats.mse
        elif 'mae' == str.lower(self.loss_name):
            self.loss_trn = self.trn_stats.mae
            if self.vld_data is not None:
                self.loss_vld = self.vld_stats.mae
            if self.tst_data is not None:
                self.loss_tst = self.tst_stats.mae
        else:
            message = 'Accepted conv stat names are: `elbo`, `llh`.'
            message += '  Received {}'.format(self.loss_name)
            raise InvalidArgumentException()
            print(message)
            raise ValueError()
        self.loss = self.loss_trn if self.vld_data is None else self.loss_vld

        # setup logging
        for stats in [self.trn_stats, self.vld_stats, self.tst_stats]:
            if stats is not None:
                stats.setup_logging(**logging_options)
        self.summarize = tf.summary.merge_all()


    def initialize(self, logdir='', rfreq=10, max_iter=1000, min_iter=30,
            epsilon=1e-2, better_than_n_ago=5, skip_init=False):
        """ Initialize variables and set up logging

        Parameters
        ----------
        logdir : str, optional
            logdir for tensorflow
        rfreq : int, optional
            Number of iterations between convergence checks, and print updates.
        max_iter : int, optional
            Maximum number of iterations to run without convergence. Default 1001.
        min_iter : int, optional
            Minimum number of iterations to run regardless of convergence. Default
            30.
        epsilon : float, optional
            percent magnitude of change for assessing convergence.  Default 0.01%
        better_than_n_ago : int, default 5
            terminate if loss worse than n*rfreq steps ago and getting worse
        skip_init : bool, optional
            Skip initialization step if True, and reopen train_writer if logging.
            Intended for use with a second call to to `run`. Note all changes in
            logging settings will be ignored, but rfreq, max_tier, min_iter, and
            epsilon may be amended.
        """
        # setup convergence parameters
        self.rfreq = rfreq
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.epsilon = epsilon
        self.better_than_n_ago = better_than_n_ago

        if skip_init:
            if self.logging:
                self.train_writer.reopen()
        else:
            # set up logging
            if logdir is not None and len(logdir) > 0:
                self.logging = True
                self.logdir = logdir
                self.train_writer = tf.summary.FileWriter(logdir,
                        tf.get_default_graph())
            else:
                self.logging = False

            init = tf.global_variables_initializer()

            sess = get_session()
            sess.run(init)


    def update(self, feed_dict):
        """ Update method

        Returns
        -------
        info_dict : dict
            information from run
        """
        sess = get_session()
        for op in self.train:
            sess.run(op, feed_dict=feed_dict)
        t = sess.run(self.increment_t)

        check_step = t % self.rfreq == 0 or t == 1
        log_step = check_step and t != 1
        loss = sess.run(self.loss, feed_dict=feed_dict) if check_step else None
        if self.logging and log_step:
            summary = sess.run(self.summarize, feed_dict=feed_dict)
            self.train_writer.add_summary(summary, self.t.eval())

        return {'t': t, 'loss': loss}


    def run(self, trn_tensorvalue, vld_tensorvalue=None, tst_tensorvalue=None,
            save_init=False, save_inflection=False, *args, **kwargs):
        """
        Parameters
        ----------
        trn_tensorvalue : tf.SparseTensorValue or tuple
            valid feeddict value for tf.sparse_placeholder.  either
            tf.SparseTensorValue or 3-tuple of (indices, values, dense_shape)
        vld_tensorvalue : tf.SparseTensorValue or tuple, optional
            valid feeddict value for tf.sparse_placeholder.  either
            tf.SparseTensorValue or 3-tuple of (indices, values, dense_shape).
            Required if self.vld_data is not None.
        tst_tensorvalue : tf.SparseTensorValue or tuple, optional
            valid feeddict value for tf.sparse_placeholder.  either
            tf.SparseTensorValue or 3-tuple of (indices, values, dense_shape).
            Required if self.tst_data is not None.
        save_init : bool, optional
            save initialization. [default False]
        save_inflection : bool, optional
            save when inflection point happens. [default False]
        *args :
            passed to ``initialize``.
        **kwargs :
            passed to ``initialize``.
        """
        self.initialize(*args, **kwargs)
        print(json.dumps(self.hyper_prm.to_dict()))

        sess = get_session()

        if save_init and self.logging:
            self.vi_prm.write_params_to_file(self.logdir, prefix='init.')

        # set up feed dict
        feed_dict = {self.trn_data : trn_tensorvalue}
        if self.vld_data is not None:
            feed_dict[self.vld_data] = vld_tensorvalue
        if self.tst_data is not None:
            feed_dict[self.tst_data] = tst_tensorvalue

        t = sess.run(self.t)
        loss, converged, wrong_way, pct_change = [], False, False, [1.0]
        while t < self.max_iter and not (converged or wrong_way) \
                or t < self.min_iter:
            info_dict = self.update(feed_dict=feed_dict)
            t, loss_t = info_dict['t'], info_dict['loss']
            if loss_t is not None:
                loss.append(loss_t)
                if len(loss) > 1:
                    delta = loss[-1] - loss[-2]
                    pct = 100 * delta /  np.abs(loss[-2])
                    converged = ((np.abs(pct) < self.epsilon) and \
                                (np.abs(pct_change[-1]) < self.epsilon)) and \
                                (np.sign(pct) <= np.sign(pct_change[-1]))
                                # try to avoid halting at - to + inflection points
                    pct_change.append(pct)

                    # assume always want to go toward zero, regardless of func
                    if len(loss) >= self.better_than_n_ago and \
                            self.better_than_n_ago > 0:
                        worse_than_n_ago = np.abs(loss[-self.better_than_n_ago]
                                ) < np.abs(loss[-1])
                        getting_worse = np.abs(loss[-2]) < np.abs(loss[-1])
                        wrong_way = worse_than_n_ago and getting_worse

                    if save_inflection and np.sign(pct_change[-1]) != \
                            np.sign(pct_change[-2]):
                        print('...saving inflection point')
                        self.vi_prm.write_params_to_file(self.logdir,
                                prefix='iteration{}.'.format(t))
                else:
                    delta, pct = np.inf, np.inf
                info_dict['delta'] = delta
                info_dict['pct_change'] = pct_change[-1]
                self.print_progress(info_dict)


        if wrong_way:
            print('wrong_way')
        self.finalize()
        return loss


    def print_progress(self, info_dict):
        """Print progress to output.
        Parameters
        ----------
        info_dict : dict
            Dictionary of algorithm-specific information.
        """
        t = info_dict['t']

        if t % self.rfreq == 0:
            loss = info_dict['loss']
            pct = '{0:.5f}'.format(info_dict['pct_change'])
            msg = 'Iteration {0}'.format(str(t).rjust(len(str(self.max_iter))))
            msg += ' : {0:.5f} [{1}%]'.format(loss, pct.rjust(9))
            print(msg)


    def finalize(self):
        """Function to call after convergence.
        """
        if self.logging:
            self.train_writer.close()


    def _build_capacity_updates(self):
        """ Build updates for capacity gamma

        Returns
        -------
        capacity_updates :
            tensorflow grouped operation for capacity gamma updates
        """
        xi_shape = CAVICalculator.get_capacity_shape_update(
                nfactors=self.hyper_prm.nfactors,
                shape_prior=self.hyper_prm.ap,
                child_shape_prior=self.hyper_prm.a)
        eta_shape = CAVICalculator.get_capacity_shape_update(
                nfactors=self.hyper_prm.nfactors,
                shape_prior=self.hyper_prm.cp,
                child_shape_prior=self.hyper_prm.c)

        xi_invrate = CAVICalculator.get_capacity_invrate_update(
                invrate_prior=self.hyper_prm.bp,
                child_e_x = self.vi_prm.theta.e_x)
        eta_invrate = CAVICalculator.get_capacity_invrate_update(
                invrate_prior=self.hyper_prm.dp,
                child_e_x = self.vi_prm.beta.e_x)

        assign_ops = self.vi_prm.xi.get_assign_ops(
                shape=xi_shape,
                invrate=xi_invrate)
        assign_ops.extend(self.vi_prm.eta.get_assign_ops(
                shape=eta_shape,
                invrate=eta_invrate))

        return tf.group(*assign_ops, name='capacity_updates')


    def _build_loading_updates(self, data):
        """ Build updates for loading gammas

        Updates are returned in a tuple, and should be run consecutively, not
        concurrently.
        """
        theta_shape, beta_shape = CAVICalculator.get_loading_shape_updates(
                theta_shape_prior=self.hyper_prm.a,
                beta_shape_prior=self.hyper_prm.c,
                data=data,
                log_phi=self.vi_prm.z.log_phi,
                nfactors=self.hyper_prm.nfactors)

        theta_invrate = CAVICalculator.get_loading_invrate_update(
                prior_e_x=self.vi_prm.xi.e_x,
                other_loading_e_x=self.vi_prm.beta.e_x)
        beta_invrate = CAVICalculator.get_loading_invrate_update(
                prior_e_x=self.vi_prm.eta.e_x,
                other_loading_e_x=self.vi_prm.theta.e_x)

        assign_ops_theta = self.vi_prm.theta.get_assign_ops(shape=theta_shape,
                invrate=theta_invrate)
        assign_ops_beta = self.vi_prm.beta.get_assign_ops(shape=beta_shape,
                invrate=beta_invrate)

        return (tf.group(*assign_ops_theta, name='theta_updates'),
                tf.group(*assign_ops_beta, name='beta_updates'))


    def _build_z_updates(self, indices):
        """ Build updates for z

        Parameters
        ----------
        indices :
            indices of observed data
        """
        log_rho = CAVICalculator.get_multinomial_unnorm_log_update(
                indices=indices,
                theta_e_logx=self.vi_prm.theta.e_logx,
                beta_e_logx=self.vi_prm.beta.e_logx)
        assign_ops = self.vi_prm.z.get_assign_ops(log_rho)
        return tf.group(*assign_ops, name='z_updates')


class CAVICalculator:
    """ Coordinate Ascent Variational Inference update calculator for HPF's
        variational parameters.

    Note
    ----
    This currently requires passing individual parameters to each function every time
    it is called as opposed to just hpf_hyper_params and hpf_vi_params objects.  It
    would perhaps be more elegant to pass these objects (or actually make this its
    own object instantiated with everything it needs or to merge it into
    HPFInference). However, I prefer explicit passing as it is clearer what each
    function actually depends upon from its method invocation.
    """

    @staticmethod
    def get_capacity_shape_update(nfactors, shape_prior, child_shape_prior):
        """
        Parameters
        ----------
        nfactors : int
            number of factors
        shape_prior : float
            shape hyperprior (ap for xi, cp for eta)
        child_shape_prior : float
            shape hyperprior for child (a for xi, c for eta)


        Returns
        -------
        new_shape : float
            updated shape for capacity

        Notes
        -----
        Only needs to be called once because set from hyperpriors alone.
        Note that this reduces the parameter's shape from an ncell or
        ngene row vector to a scalar, as the value is shared.
        """
        return shape_prior + nfactors * child_shape_prior


    @staticmethod
    def get_capacity_invrate_update(invrate_prior, child_e_x):
        """
        Parameters
        ----------
        invrate_prior : float
            inverse rate hyperprior (ap/bp for xi, cp/dp for eta)
        child_e_x : tensor
            expected value of variational child (theta.e_x for xi,
            beta.e_x for eta).  These are (ncell, nfactor) and (ngene, nfactor)
            shaped tensors for xi and eta, respectively.

        Returns
        -------
        new_invrate : tensor
            updated inverse rates for capacity variational distribution
        """
        return invrate_prior + tf.reduce_sum(child_e_x, axis=1, keep_dims=True)


    @staticmethod
    def get_loading_invrate_update(prior_e_x, other_loading_e_x):
        """
        Parameters
        ----------
        prior_e_x : tensor
            expected values of the variational parent capacity. xi.e_x for theta and
            eta.e_x for beta. An (ncell,1) or (ngene,1) shaped tensor, respectively.
        other_loading_e_x : tensor
            expected value of the other variational loading.  beta.e_x for theta and
            theta.e_x for beta. An (ncell, nfactor) or (ngene, nfactor) shaped tensor
            respectively.

        Returns
        -------
        new_invrate : tensor
        """
        return prior_e_x + tf.reduce_sum(other_loading_e_x, axis=0, keep_dims=True)


    @staticmethod
    def get_loading_shape_updates(theta_shape_prior, beta_shape_prior, data, log_phi,
            nfactors):
        """
        Parameters
        ----------
        theta_shape_prior: float
            hyperprior for theta HPFGamma shape (hyperparameter a)
        beta_shape_prior: float
            hyperprior for beta HPFGamma shape (hyperparameter c)
        data: tensoflow sparsetensor
            data
        log_phi: tensor
            multinomial variational distribution for auxiliary variables.  shaped
            like: (nsamples, nfactors)

        Returns
        -------
        new_theta_shape : tensor
            new shape for theta
        new_beta_shape : tensor
            new shape for beta

        Notes
        -----
        Calculate theta and beta's shape updates together because they both require
        y_ui * phi_uik, just summed over different axes, and this way we don't have
        to calculate it twice.  Additionally, they do not depend directly on each
        other's expectations, so it is valid to calculate them simultaniously.
        """
        sum_over_cells, sum_over_genes = [], []
        nfactors = log_phi.get_shape().as_list()[1]
        unstacked = tf.unstack(log_phi, axis=1, num=nfactors, name='log_phi_k')
        for log_phi_k in unstacked:
            prod_k_values = tf.exp(log_phi_k) * data.values
            try:
                prod_k = tf.SparseTensor(indices=data.indices, values=prod_k_values,
                        shape=data.shape)
            except (TypeError, AttributeError):
                prod_k = tf.SparseTensor(indices=data.indices, values=prod_k_values,
                        dense_shape=data.dense_shape)
            sum_over_cells.append(tf.sparse_reduce_sum(prod_k,axis=0))
            sum_over_genes.append(tf.sparse_reduce_sum(prod_k,axis=1))

        # theta_update should be a (ncell,nfactor) tensor
        theta_update = tf.squeeze(tf.stack(sum_over_genes, axis=1))
        # beta_update should be a (ngene,nfactor) tensor
        beta_update = tf.squeeze(tf.stack(sum_over_cells, axis=1))

        return [theta_shape_prior+theta_update, beta_shape_prior+beta_update]


    @staticmethod
    def get_multinomial_unnorm_log_update(indices, theta_e_logx, beta_e_logx):
        """ Get update for auxiliary variables: unnnormalized multinomial, rho

        Parameters
        ----------
        indices : tensor
            data indices
        theta_e_logx : tensor
            HPFGamma.e_logx for cell loadings
        beta_e_logx : tensor
            HPFGamma.e_logx for gene loadings
        """
        indices = tf.cast(indices, tf.int32)
        thetas = tf.gather(theta_e_logx, indices[:,0])
        betas = tf.gather(beta_e_logx, indices[:,1])
        updates =  thetas + betas
        return updates
