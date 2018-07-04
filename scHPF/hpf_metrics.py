#!/usr/bin/python

import numpy as np
import tensorflow as tf

from util import get_median_1d


class HPFMetrics(object):
    """ Object containing metrics for a given dataset and variational params
    """
    def __init__(self, vi_prm, data, data_name='train', elbo_calc_phi=False):
        self.vi_prm = vi_prm
        self.data = data
        self.data_name = data_name

        mkname = lambda x : x + '_' + self.data_name

        with tf.name_scope(data_name + '_stats'):
            # ELBO
            self.elbo, components = self.vi_prm.elbo(self.data,
                    as_components=True, calc_phi=elbo_calc_phi)
            self.elbo_xi = components[0]
            self.elbo_eta = components[1]
            self.elbo_theta = components[2]
            self.elbo_beta = components[3]
            self.elbo_pois = components[4]

            # LLH variations
            self.llhp = self.vi_prm.log_likelihood(self.data,
                    weighted=False, pointwise=True, name=mkname('llhp'))
            self.llhp_valmin = self.data.values[tf.argmin(self.llhp)]
            self.llh_mean = tf.reduce_mean(self.llhp, name=mkname('llh'))
            self.llh_median = get_median_1d(self.llhp)
            self.llhw_trn = self.vi_prm.log_likelihood(self.data,
                    weighted=True, name=mkname('llhw'))
            self.llh_complete = self.vi_prm.log_likelihood_complete(
                    self.data, name=mkname('llh_complete'))

            # residual based
            self.mae = self.vi_prm.mae(self.data,
                    e_poisson_rate=self.vi_prm.poisson_rate)
            self.mse = self.vi_prm.mse(self.data,
                    e_poisson_rate=self.vi_prm.poisson_rate)
            self.residual = self.vi_prm.residual(self.data,
                    as_sparse=False)
            self.varexp = self.vi_prm.total_variation_explained(
                    self.data, residual=self.residual)
            self.s2exp =  self.vi_prm.total_variance_explained(
                    self.data, residual=self.residual)


    def setup_logging(self, log_elbo=False, log_elbo_components=False, log_llhw=False,
            log_llh=True, log_mse=False, log_mae=True, log_xi=False, log_eta=False,
            log_theta=False, log_beta=False, log_phi=True, log_llhp=False):
        """

        """
        mkname = lambda x : x + '/' + self.data_name

        if log_elbo:
            tf.summary.scalar(mkname('elbo'), self.elbo)
            if log_elbo_components:
                tf.summary.scalar(mkname('elbo') + '_xi', self.elbo_xi)
                tf.summary.scalar(mkname('elbo') + '_eta', self.elbo_eta)
                tf.summary.scalar(mkname('elbo') + '_theta', self.elbo_theta)
                tf.summary.scalar(mkname('elbo') + '_beta', self.elbo_beta)
                tf.summary.scalar(mkname('elbo') + '_pois', self.elbo_pois)
        if log_llh:
            tf.summary.scalar(mkname('llh_mean'), self.llh_mean)
            tf.summary.scalar(mkname('llh_median'), self.llh_median)
            tf.summary.scalar(mkname('llh_complete'), self.llh_complete)
        if log_llhw:
            tf.summary.scalar(mkname('llhw'), self.llhw)
        if log_llhp:
            tf.summary.histogram(mkname('llhp'), self.llhp)
            tf.summary.scalar(mkname('llhp_valmin'), self.llhp_valmin)
        if log_mae:
            tf.summary.scalar(mkname('mae'), self.mae)
        if log_mse:
            tf.summary.scalar(mkname('mse'), self.mse)
        if log_xi:
            tf.summary.histogram('xi_invrate', self.vi_prm.xi.invrate)
        if log_eta:
            tf.summary.histogram('eta_invrate', self.vi_prm.eta.invrate)
        if log_theta:
            for k in range(self.hyper_prm.nfactors):
                tf.summary.histogram('theta_e_x/%02d' % (k),
                        self.vi_prm.theta.e_x[:,k])
        if log_beta:
            for k in range(self.hyper_prm.nfactors):
                tf.summary.histogram('beta_e_x/%02d' % (k),
                        self.vi_prm.beta.e_x[:,k])
        if log_phi:
            for k in range(self.hyper_prm.nfactors):
                tf.summary.histogram('phi/%02d' % (k),
                        tf.exp(self.vi_prm.z.log_phi[:,k]))


