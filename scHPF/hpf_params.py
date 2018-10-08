#!/usr/bin/python

import os
import datetime
from pickle import UnpicklingError
import yaml

import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Poisson as tfPoisson

from .util import get_session, log_likelihood_poisson
from .hpf_hyperparams import HyperParams
from .hpf_inference import CAVICalculator


class HPFGamma:
    """ Gammas (inverse rate parameterization) for loadings and capacities.

    Notes
    -----
    * Randomized start values of capacities are never used if updated
      before loading parameters theta and beta are updated.
    * This class is modeled after hpf_gamma in Francisco Ruiz's hpf C++ code
    """

    @staticmethod
    def init_random(shape_prior, invrate_prior, shape, name='gamma',
            dtype=tf.float64):
        """ Initialize HPFGamma with a size shaped shape hyperparameter

        Parameters
        ----------
        shape_prior : float
            seed for setting shape of each independent gamma
        invrate_prior : flost
            seed for setting inverse rate of each independent gamma
        shape : tuple
            shape of matrix of gammas (numpy style shape)
        dtype : tensorflow dtype
            tensorflow dtype of parameters and expectations
        name : str
            name of the variable

        Returns
        -------
        hpf_gamma : HPFGamma
            HPFGamma instance with shape and inverse rate parameters randomly
            initialized to values between 0.5x and 1.5x the given seed.  Corresponds
            to variational parameters of each latent gamma-distributed variable.
        """
        shape_init = tf.random_uniform(minval=0.5*shape_prior,
                maxval=1.5*shape_prior, shape=shape, dtype=dtype)
        invrate_init = tf.random_uniform(minval=0.5*invrate_prior,
                maxval=1.5*invrate_prior, shape=shape, dtype=dtype)
        return HPFGamma(shape_init, invrate_init, dtype=dtype, name=name)


    @staticmethod
    def init_random_constant_shape(shape_prior, invrate_prior, shape,
            name='hgamma', dtype=tf.float64):
        """ Initialize HPFGamma with a scalar shape
        Parameters
        ----------
        shape_prior : float
            seed for setting shape of each independent gamma, with shape
            shared by all gammas.
        invrate_prior : float
            seed for setting inverse rate of each independent gamma
        shape : tuple
            shape of matrix of gammas (numpy style shape)
        dtype : tensorflow dtype
            tensorflow dtype of parameters and expectations
        name : str
            name of the variable

        Returns
        -------
        hpf_gamma : HPFGamma
            HPFGamma instance with shape and inverse rate parameters randomly
            initialized to values between 0.5x and 1.5x the given seed.
        """
        shape_init = tf.random_uniform(minval=0.5*shape_prior,
                maxval=1.5*shape_prior, shape=[], dtype=dtype)
        invrate_init = tf.random_uniform(minval=0.5*invrate_prior,
                maxval=1.5*invrate_prior, shape=shape, dtype=dtype)
        return HPFGamma(shape_init, invrate_init, dtype=dtype, name=name)


    @staticmethod
    def load_from_file(shape_file, invrate_file, name='gamma', dtype=tf.float64):
        """ Load a gamma from file
        Parameters
        ----------
        shape_file : str
            file containing gamma shape variational parameters
        invrate_file : float
            file containing gamma inverse rate variational parameters
        dtype : tensorflow dtype
            tensorflow dtype of parameters and expectations
        name : str
            name of the variable

        Returns
        -------
        hpf_gamma : HPFGamma
            HPFGamma instance with shape and inverse rate parameters loaded from
            files
        """
        try:
            shape_init = np.load(shape_file)
            invrate_init = np.load(invrate_file)
        except UnpicklingError:
            shape_init = np.loadtxt(shape_file)
            invrate_init = np.loadtxt(invrate_file)
        return HPFGamma(tf.constant(shape_init, dtype=dtype),
                tf.constant(invrate_init, dtype=dtype), name=name)


    def __init__(self, shape_init, invrate_init, name='gamma', dtype=tf.float64):
        """
        Parameters
        ----------
        shape_init : tf initializer
            matrix of shapes for independent gammas
        invrate : tf initializer
            matrix of inverse rates for independent gammas
        name : str
            name of the variable (scope for all encapsulated tensors)
        dtype : tensorflow dtype
            tensorflow dtype of parameters and expectations
        """
        self.name = name
        self.dtype = dtype
        with tf.variable_scope(name):
            # variables
            self.shape = tf.get_variable(name='shape', initializer=shape_init,
                    trainable=False)
            self.invrate = tf.get_variable(name='invrate', trainable=False,
                    initializer=invrate_init)

            with tf.name_scope(name + '_ops'):
                # expectations
                self.e_x = tf.divide(self.shape, self.invrate, name='e_x')
                self.e_logx = tf.subtract(tf.digamma(self.shape),
                        tf.log(self.invrate), name='e_logx')

                # entropy
                self.entropy = tf.add(self.shape - tf.log(self.invrate),
                    tf.lgamma(self.shape) + (1 - self.shape)*tf.digamma(self.shape),
                    name='entropy')

                # transformations, only meaningful for non-hierarchical priors
                self.norm = tf.divide(self.e_x, tf.reduce_sum(self.e_x, 1,
                    keep_dims=True), name='norm')
                k = self.e_x.get_shape().as_list()[1]
                geo_mean = tf.pow(tf.reduce_prod(self.e_x, 1, keep_dims=True), 1/k,
                    name='geomean')
                self.geonorm = tf.divide(self.e_x, geo_mean, name='geonorm')
                self.termscore = tf.multiply(self.norm, tf.log(self.geonorm),
                    name='termscore')


    def get_assign_ops(self, shape, invrate):
        """ Get list of tensorflow update operations
        Parameters
        ----------
        shape : tensorflow tensor
            new shape parameter for gamma
        invrate : tensorflow tensor
            new inverse rate parameter for gamma
            False.

        Returns
        -------
        updates : list
            list of new values for gamma's shape, invrate, e_x, and e_logx.
            Values should be grouped using tf.group before being referenced
            again.
        """
        assign_ops = []
        assign_ops.append(self.shape.assign(shape))
        assign_ops.append(self.invrate.assign(invrate))
        return assign_ops


    def sample(self, nsamples=1, name=None):
        """Sample from a gamma distributions
        Parameters
        ----------
        nsamples : int, optional
            Number of samples to take from each gamma
        name : str, optional
            optional name for operation

        Returns
        -------
        samples : tensor
            [nsamples, self.invrate.shape[0], self.invrate.shape[1]] tensor of
            samples
        """
        return tf.random_gamma(shape=[nsamples], alpha=self.shape,
                beta=self.invrate, dtype=self.dtype, name=name)


class HPFMultinomial:
    """ Multinomial reflecting assignments of molecules across factors for each
        nonzero entry
    Notes
    -----
    * Randomized start values are never used if the multinomial is
      updated (via hpf_cavi_updater + calling get_updater) before loading
      parameters theta and beta are updated.
    * This class is modeled after hpf_multinomial in Francisco Ruiz's hpf C++
      code
    """

    @staticmethod
    def init_random_symmetric(nsamples, nfactors, name='z', dtype=tf.float64):
        """ Initialize normalized loadings by drawing from a symmetric dirichlet

        Parameters
        ----------
        nsamples : int
            number of datapoints sampled. (entries in test set)
        nfactors : int
            number of factors (K for dirichlet)
        name : str
            name of the variable (scope for all encapsulated tensors)
        dtype : tensorflow dtype
            dtype of tensor.  default tf.float64

        Returns
        -------
        HPFMultinomial: HPFMultinomial
            Variational distribution of assignments of molecules across factors
            for each nonzero entry in test set
        """
        return HPFMultinomial.init_random(nsamples=nsamples,
                dirichlet_prior=np.ones(nfactors), name=name,
                dtype=dtype)


    @staticmethod
    def init_random(nsamples, dirichlet_prior, name='z', dtype=tf.float64):
        """ Initialize normalized loadings by drawing from a dirichlet
            specified by an arbitrary prior

        Parameters
        ----------
        nsamples : int
            number of datapoints sampled. (entries in test set)
        dirichlet_prior : numpy array
            prior for dirichlet
        name : str
            name of the variable (scope for all encapsulated tensors)
        dtype : tensorflow dtype
            dtype of tensor.  default tf.float64

        Returns
        -------
        HPFMultinomial : HPFMultinomial
            Variational distribution of assignments of molecules across factors
            for each nonzero entry in test set
        """
        def get_random_ldirichlet():
            return np.log(np.random.dirichlet(dirichlet_prior, nsamples) + 1e-100)
        ldirichlet_init = tf.py_func(get_random_ldirichlet, inp=[],
                Tout=[tf.double])[0]
        ldirichlet_init.set_shape((nsamples, dirichlet_prior.shape[0]))
        init = tf.saturate_cast(ldirichlet_init, dtype)
        return HPFMultinomial(init, dtype=dtype, name=name)


    @staticmethod
    def load_from_file(log_rho_file='', name='z', dtype=tf.float64):
        """ Load auxiliary variables from file file(s). Formats readable by np.load
            or np.loadtxt valid
        Parameters
        ----------
        log_rho : tensor
            file of log(unnormalized loadings over factors per entry).
            Should have shape (nfactor, nsamples), where nsamples is the number of
            nonzero entries (the training data).
        name : str
            name of the variable (scope for all encapsulated tensors)
        dtype : tensorflow dtype
            dtype of tensor.  default tf.float64

        Returns
        -------
        hpf_multinomial : HPFMultinomial
            Variational distribution of assignments of molecules across factors
            for each nonzero entry in test set, loaded from file.
        """
        try:
            log_rho = np.load(log_rho_file)
        except UnpicklingError:
            log_rho = np.loadtxt(log_rho_file)
        return HPFMultinomial(log_rho_init=tf.constant(log_rho, dtype=dtype),
                dtype=dtype, name=name)


    def __init__(self, log_rho_init, indices=None, name='z', dtype=tf.float64):
        """
        Parameters
        ----------
        log_rho_init : tensorflow initializer
            log( unnormalized loadings over factors per entry ). Should have shape
            (nfactors, nsamples).
        indices : tensorflow initializer, optional
            indices of samples
        name : str
            name of the variable (scope for all encapsulated tensors)
        dtype : tensorflow dtype
            dtype of tensor.  default tf.float64
        """
        self.name = name
        self.dtype = dtype
        with tf.variable_scope(name):
            self.log_rho = tf.get_variable(name='log_rho', initializer=log_rho_init,
                    trainable=False)
            self.log_phi = tf.subtract(self.log_rho, tf.reduce_logsumexp(
                self.log_rho, axis=1, keep_dims=True), name='log_phi')
            self.indices = indices


    def get_assign_ops(self, log_rho):
        """ Get list of tensorflow update operations, updating values rho, and
            calculating and updating phi as normalied rho.

        Parameters
        ----------
        log_rho : tensor
            new values for log_rho

        Returns
        -------
        updates : list
            list of new values for multinomial, both normalized and
            unnormalized.  Values should be grouped using tf.group before being
            referenced again.
        """

        assign_ops = []
        assign_ops.append(self.log_rho.assign(log_rho))
        return assign_ops


class VariationalParams:
    """
    Composite of variational parameters on which inference is performed, and
    functions for evaluation both with and without data.
    """

    def init_random(hyper_p, nsamples=0, phi_prior_init=[]):
        """ Create instance with Initialize parameters with random offsets from
            hyperpriors

        Parameters
        ----------
        hyper_p : HyperParams
            Hyperparameters of model
        nsamples : int, optional
            number of samples for training.  auxiliary variables not initialized if
            set less than 1 or not given.
        phi_prior_init : list of numpy array, optional
            dirichlet prior for auxiliary variable initialization. Length must match
            nfactors.  Symmetric dirichlet used if not given.


        Returns
        -------
        vi_params : VariationalParams
        """
        init_a = hyper_p.a0
        init_ap = hyper_p.ap0
        init_bp = hyper_p.bp0
        init_c = hyper_p.c0
        init_cp = hyper_p.cp0
        init_dp = hyper_p.dp0

        # initialize capacity parameters xi and eta
        with tf.variable_scope('variational_params'):
            xi = HPFGamma.init_random_constant_shape(
                            shape=[hyper_p.ncells,1],
                            shape_prior=init_ap,
                            invrate_prior=init_bp,
                            dtype=hyper_p.dtype,
                            name='xi')
            eta = HPFGamma.init_random_constant_shape(
                            shape=[hyper_p.ngenes,1],
                            shape_prior=init_cp,
                            invrate_prior=init_dp,
                            dtype=hyper_p.dtype,
                            name='eta')

            # initialize loading parameters theta and beta
            theta = HPFGamma.init_random(
                            shape=[hyper_p.ncells, hyper_p.nfactors],
                            shape_prior=init_a,
                            invrate_prior=init_ap / init_bp,
                            dtype=hyper_p.dtype,
                            name='theta')

            beta = HPFGamma.init_random(
                            shape=[hyper_p.ngenes, hyper_p.nfactors],
                            shape_prior=init_c,
                            invrate_prior=init_cp / init_dp,
                            dtype=hyper_p.dtype,
                            name='beta')

            if nsamples is not None and nsamples > 0:
                if len(phi_prior_init) == hyper_p.nfactors:
                    z = HPFMultinomial.init_random_asymmetric(
                            nsamples=nsamples,
                            dirichlet_prior=phi_prior_init,
                            dtype=hyper_p.dtype)
                else:
                    z = HPFMultinomial.init_random_symmetric(
                            nsamples=nsamples,
                            nfactors=hyper_p.nfactors,
                            dtype=hyper_p.dtype)
            else:
                z = None

            return VariationalParams(hyper_p=hyper_p, xi=xi, eta=eta,
                    theta=theta, beta=beta, z=z)


    @staticmethod
    def load_from_file(indir, hyper_p=None, prefix='', include_z=False,
            dtype=tf.float64, npy=True):
        """ Load variational parameters from file
        Parameters
        ----------
        indir : str
            directory containing variational parameters
        hyper_p : HyperParameters, optional
            Hyperparameters for model
        prefix : str, optional
            prefix for all varational parameters
        include_z : bool, optional
            Load auxiliary variables as well
        npy : bool, optional
            Is file stored in npy format.
        dtype : tensorflow dtype
            dtype of tensor.  default tf.float64

        Returns
        -------
        vi_params : VariationalParams

        Notes
        -----
        Filenames must be of the form : `{indir}/{prefix}theta_shape.tsv`, etc.
        """
        if hyper_p is None:
            hprefix = indir + '/' + prefix
            try :
                hyper_p = HyperParams.load_from_file(hprefix+"hyperparams.yaml",
                        dtype=dtype)
            except (KeyError, FileNotFoundError):
                hyper_p = HyperParams.load_from_file(hprefix+"run_info.yaml",
                        dtype=dtype)

        ext = 'npy' if npy else 'txt'
        instring = '{0}/{1}{2}.{3}'.format(indir, prefix, '{}',ext)

        with tf.variable_scope('variational_params'):
            xi_shape_file = instring.format('xi_shape')
            xi_invrate_file = instring.format('xi_invrate')
            xi = HPFGamma.load_from_file(shape_file=xi_shape_file,
                    invrate_file=xi_invrate_file, name='xi', dtype=dtype)

            eta_shape_file = instring.format('eta_shape')
            eta_invrate_file = instring.format('eta_invrate')
            eta = HPFGamma.load_from_file(shape_file=eta_shape_file,
                    invrate_file=eta_invrate_file, name='eta', dtype=dtype)

            theta_shape_file = instring.format('theta_shape')
            theta_invrate_file = instring.format('theta_invrate')
            theta = HPFGamma.load_from_file(shape_file=theta_shape_file,
                    invrate_file=theta_invrate_file, name='theta', dtype=dtype)


            beta_shape_file = instring.format('beta_shape')
            beta_invrate_file = instring.format('beta_invrate')
            beta = HPFGamma.load_from_file(shape_file=beta_shape_file,
                    invrate_file=beta_invrate_file, name='beta', dtype=dtype)

            log_phi_file = instring.format('log_phi')
            log_rho_file = instring.format('log_rho')
            if os.path.exists(log_rho_file) and include_z:
                z = HPFMultinomial.load_from_file( log_rho_file=log_rho_file,
                        name='z', dtype=dtype)
            else:
                z = None

            return VariationalParams(hyper_p=hyper_p, xi=xi, eta=eta,
                    theta=theta, beta=beta, z=z)


    def __init__(self, hyper_p, xi, eta, theta, beta, z=None):
        """
        Parameters
        ----------
        hyper_p : HyperParams
            Hyperparameters of model
        xi : HPFGamma
            variational parameters cell capacities
        eta : HPFGamma
            variational parameters for gene capacities
        theta : HPFGamma
            variational parameters for cell loadings
        beta : HPFGamma
            variational parameters for gene loadings
        z : HPFMultinomial,  optional
            Variational parameters for auxiliary variables.
        """
        self.hyper_p = hyper_p
        self.xi = xi
        self.eta = eta
        self.theta = theta
        self.beta = beta
        self.z = z

        with tf.name_scope('vp_ops'):
            # normalization
            self.xinorm =  tf.multiply(self.xi.e_x, self.theta.e_x,
                    name='xinorm')
            self.etanorm =  tf.multiply(self.eta.e_x, self.beta.e_x,
                    name='etanorm')

            # poisson rates
            self.poisson_rate = tf.matmul(self.theta.e_x, self.beta.e_x,
                    transpose_b=True, name='poisson_rate')
            with tf.name_scope('poisson_rate_factor'):
                theta_expand = tf.expand_dims(tf.transpose(self.theta.e_x), 1)
                beta_expand = tf.expand_dims(tf.transpose(self.beta.e_x), 2)
                self.poisson_rate_factor = tf.transpose(tf.matmul(beta_expand,
                    theta_expand), name='poisson_rate_factor')

            # adjusted poisson rates
            self.adj_poisson_rate = tf.matmul(self.xinorm, self.etanorm,
                    transpose_b=True, name='adj_poisson_rate')
            with tf.name_scope('adj_poisson_rate_factor'):
                xinorm_expand = tf.expand_dims(tf.transpose(self.xinorm), 1)
                etanorm_expand = tf.expand_dims(tf.transpose(self.etanorm), 2)
                self.adj_poisson_rate_factor = tf.transpose(tf.matmul(
                    etanorm_expand, xinorm_expand),
                    name='adj_poisson_rate_factor')


    def generate_replicate_dataset(self, nrep=1, resample_loadings=True,
            e_poisson_rate=None, name='sample'):
        """Generate replicate datasets from the posterior predictive
        Parameters
        ----------
        nrep : int, optional
            Number of replicate datasets to return. [default 1]
        resample_loadings : bool, optional
            If true, resample factor loadings and ignore `e_poisson_rate`.
            If false, use `e_poisson_rate` if passed as argument or
            self.e_poisson_rate if not. [default True]


        """
        with tf.name_scope('sample'):
            if resample_loadings:
                theta_samples = self.theta.sample(nrep)
                beta_samples = self.beta.sample(nrep)
                pois_rate_samples = tf.matmul(beta_samples, theta_samples,
                        transpose_b=True, name='poisson_rate_samples')
                p = tfPoisson(pois_rate_samples)
                return tf.transpose(p.sample(name=name))
            else:
                if e_poisson_rate is None:
                    e_poisson_rate = self.poisson_rate
                p = tfPoisson(tf.transpose(e_poisson_rate))
                return tf.transpose(p.sample(nrep, name=name))


    def residual(self, data, e_poisson_rate=None, name='residual',
            as_sparse=False):
        """Data residual given poisson rate

        Parameters
        ----------
        data : tensorflow SparseTensor
            data to test
        e_poisson_rate: tensor, optional
            Matrix of predictions (for reuse if already calculated).  If not given,
            calculated using self.poisson_rate
        name: name for returned tensor

        Returns
        --------
        residual: tensor or SparseTensor
        """
        with tf.name_scope('residual'):
            if e_poisson_rate is None:
                e_poisson_rate = self.poisson_rate
            e_poisson_known = tf.gather_nd(params=e_poisson_rate,
                                        indices=data.indices)
            asis = tf.equal(tf.size(tf.shape(e_poisson_known)), 1)
            f1 = lambda : tf.saturate_cast(data.values,e_poisson_known.dtype)
            f2 = lambda : tf.expand_dims( tf.saturate_cast(data.values,
                e_poisson_known.dtype) , 1)
            d = tf.cond(asis, f1, f2)
            rval = tf.subtract(d, e_poisson_known, name='rval')
            if as_sparse:
                return tf.SparseTensor(indices=data.indices, values=rval,
                        dense_shape=data.dense_shape)
            else:
                return rval


    def total_variation_explained(self, data, residual=None,
            poisson_rate=None, name='total_variation_explained'):
        """Variance explained given Poisson rate
        Defined as 1 - MSE/VAR[Data]

        Parameters
        ----------
        data : tensorflow SparseTensor, optional
            data to test.
        residual : tensor, optional
            Matrix of residuals (for reuse if already calculated). If not given,
            calculated using self.residual
        poisson_rate: tensor, optional
            Matrix of predictions (for reuse if already calculated). If not given
            and residual not given, calculated using self.poisson_rate
        name: str, optional
            name for returned tensor

        Returns
        --------
        var_exp: SparseTensor
        """
        with tf.name_scope('total_variation_explained'):
            if residual is None:
                if poisson_rate is None:
                    poisson_rate = self.poisson_rate
                r = self.residual(data, poisson_rate, as_sparse=False)
            else:
                r = residual
            mse = tf.reduce_mean(tf.pow(r, 2))
            vardata = tf.nn.moments(data.values, axes=[0])[1]
            var_unexplained = tf.divide(mse, vardata)
            var_exp = tf.subtract(tf.constant(1.0,dtype=var_unexplained.dtype),
                    var_unexplained, name=name)
            return var_exp


    def total_variance_explained(self, data, residual=None,
            poisson_rate=None, name='total_variance_explained'):
        """Second order variation explained given Poisson rate and or residual
        Defined as 1 - VAR[RESIDUAL]/VAR[Data]

        Parameters
        ----------
        data : tensorflow SparseTensor, optional
            data to test.
        residual : tensor, optional
            Matrix of residuals (for reuse if already calculated). If not given,
            calculated using self.residual
        poisson_rate: tensor, optional
            Matrix of predictions (for reuse if already calculated). If not given
            and residual not given, calculated using self.poisson_rate
        name: str, optional
            name for returned tensor

        Returns
        --------
        var_exp: SparseTensor
        """
        with tf.name_scope('total_variance_explained'):
            if residual is None:
                if poisson_rate is None:
                    poisson_rate = self.poisson_rate
                r = self.residual(data, poisson_rate, as_sparse=False)
            else:
                r = residual
            varresid = tf.nn.moments(r, axes=[0])[1]
            vardata = tf.nn.moments(data.values, axes=[0])[1]
            var_unexplained = tf.divide(varresid, vardata)
            var_exp = tf.subtract(tf.constant(1.0,dtype=var_unexplained.dtype),
                    var_unexplained, name=name)
            return var_exp


    def factor_variation_explained(self, data, residual=None,
            poisson_rate=None, poisson_rate_factor=None,
            name='factor_variation_explained'):
        """Variance explained given Poisson rate per factor.
        Calculated as (SSE_minus - SSE_tot) / SSE_minus, where SSE_tot is the
        sum of squared error given the expected poisson rate, and SSE_minus is
        the sum of squred error give the expected poisson rate *minus* the
        contribution of the factor.

        Parameters
        ----------
        data : tensorflow SparseTensor, optional
            data to test. Not needed if residual given
        residual : tensor, optional
            Matrix of residuals (for reuse if already calculated). If not given,
            calculated using self.residual
        poisson_rate: tensor, optional
            Matrix of predictions (for reuse if already calculated). If not given
            and residual not given, calculated using self.poisson_rate
        data : tensorflow SparseTensor
            data to test
        poisson_rate: tensor, optional
        poisson_rate_factor: tensor, optional
        name: str, optional
            name for returned tensor

        Returns
        --------
        var_exp: SparseTensor
        """
        with tf.name_scope('factor_variation_explained'):
            if residual is None:
                if poisson_rate is None:
                    poisson_rate = self.poisson_rate
                r = self.residual(data, poisson_rate, as_sparse=False)
            else:
                r = residual
            if poisson_rate_factor is None:
                poisson_rate_factor = self.poisson_rate_factor
            sum_sq_error_tot = tf.reduce_sum(tf.pow(r, 2))
            # poisson rate excluding contribution of factor
            rate_minus_f = tf.subtract(tf.expand_dims(poisson_rate,2),
                    poisson_rate_factor, name='rate_minus_factor')
            r_minus_f = self.residual(data, rate_minus_f)
            sum_sq_error_minus_f = tf.reduce_sum(tf.pow(r_minus_f,2), 0)
            var_exp_f = tf.divide(sum_sq_error_minus_f - sum_sq_error_tot,
                    sum_sq_error_minus_f, name=name)
            return var_exp_f


    def dominance_at_p(self, data, p=0.8, poisson_rate_factor=None,
            name='dominance_p'):
        """ Dominance of a factor at fraction p
        Sum(factor's contribution to entries where it accounts for >p faction of
        expression in that entry) / sum(factor's contribution to all entries).

        Parameters
        ----------
        data : tensorflow SparseTensor
            data over which dominance is calculated
        p : float, optional
            fraction at which to calculate dominace
        poisson_rate_factor: tensor, optional
            contribution of each factor

        Returns
        -------
        dominance_at_p : tensor
        """
        if p is None or p < 0 or p > 1:
            # TODO: exception
            print('Invalid p')
            return
        with tf.name_scope('dominance'):
            if poisson_rate_factor is None:
                poisson_rate_factor = self.adj_poisson_rate_factor
            rate_known = tf.gather_nd(params=poisson_rate_factor,
                                      indices=data.indices)
            rate_known_norm = tf.divide(rate_known, tf.reduce_sum(rate_known,
                axis=1, keep_dims=True))
            num = tf.reduce_sum(tf.cast(rate_known_norm > p, tf.float64)
                    *rate_known, axis=0)
            denom = tf.reduce_sum(rate_known, axis=0)
            return tf.divide(num, denom, name=name)


    def log_likelihood(self, data, e_poisson_rate=None, weighted=False,
            pointwise=False, name='llh'):
        """ Compute the log likelihood of the passed data under either the current
            variational distribution or the passed poisson rate

        Parameters
        ----------
        data : tensorflow SparseTensor
            data to test
        e_poisson_rate : tensor, optional
            Matrix of predictions (for reuse if already calculated).  If not given,
            calculated using self.poisson_rate
        weighted : bool, optional
            Weight loglikelihoods by the inverse of the data
        pointwise : bool, optional
            return pointwise loglikelihood
        name : str, optional
            name_scope for ops

        Returns
        -------
        llh : tensor
            reduced mean
        """
        with tf.name_scope(name):
            if e_poisson_rate is None:
                e_poisson_rate = self.poisson_rate
            e_poisson_known = tf.gather_nd(params=e_poisson_rate,
                                        indices=data.indices[:,:2])
            llh = log_likelihood_poisson(e_poisson_known, data.values,
                    name='llh_pointwise')
            if pointwise and not weighted:
                return llh
            elif pointwise and weighted:
                return tf.divide(llh, data.values,
                        name='llh_invweight')
            elif not pointwise and not weighted:
                return tf.reduce_mean(llh, name='llh')
            else:
                return tf.reduce_mean(llh / data.values,
                        name='llh_invweight')


    def log_likelihood_complete(self, data, name='llh_complete'):
        """ Compute the log likelihood of the passed data assuming zeros for all
            other entries.

        Parameters
        ----------
        data : tensorflow SparseTensor
            data to test

        Returns
        -------
        llh : tensor
            full llh, assuming all other zeros
        """
        with tf.name_scope(name):
            e_poisson_known = tf.gather_nd(params=self.poisson_rate,
                                        indices=data.indices[:,:2])
            data_terms = data.values * tf.log(e_poisson_known) - \
                    tf.lgamma(data.values + tf.constant(1.0, dtype=data.dtype))
            theta_sums = tf.reduce_sum(self.theta.e_x, axis=0, keep_dims=True)
            beta_sums = tf.reduce_sum(self.beta.e_x, axis=0, keep_dims=True)
            rate_sums = tf.squeeze(tf.matmul(theta_sums, beta_sums,
                transpose_b=True))
            return tf.reduce_sum(data_terms) - rate_sums


    def mae(self, data, e_poisson_rate=None):
        """ Compute the mean squared error of the passed data under either the
            current variational distribution or the passed poisson rate
            (e_poisson_rate)

        Parameters
        ----------
        data : tensorflow SparseTensor
            data to mean squared error of
        e_poisson_rate : tensorflow tensor, optional
            tensor of predictions (for reuse if already calculated) if not given,
            uses self.poisson_rate to calculated

        Returns
        -------
        mse : tensor
        """
        with tf.name_scope('mae'):
            if e_poisson_rate is None:
                e_poisson_rate = self.poisson_rate
            e_poisson_known = tf.gather_nd(params=e_poisson_rate,
                                        indices=data.indices[:,:2])
            mae = tf.abs(e_poisson_known - data.values, name='mae_pointwise')
            return tf.reduce_mean(mae, name='mae')


    def mse(self, data, e_poisson_rate=None):
        """ Compute the mean squared error of the passed data under either the
            current variational distribution or the passed poisson rate
            (e_poisson_rate)

        Parameters
        ----------
        data : tensorflow SparseTensor
            data to mean squared error of
        e_poisson_rate : tensorflow tensor, optional
            tensor of predictions (for reuse if already calculated) if not given,
            uses self.poisson_rate to calculated

        Returns
        -------
        mse : tensor
        """
        with tf.name_scope('mse'):
            if e_poisson_rate is None:
                e_poisson_rate = self.poisson_rate
            e_poisson_known = tf.gather_nd(params=e_poisson_rate,
                                        indices=data.indices[:,:2])
            mse = tf.square(e_poisson_known - data.values, name='mse_pointwise')
            return tf.reduce_mean(mse, name='mse')


    def elbo(self, data, calc_phi=False, as_components=False):
        """ Approximates the ELBO
        Parameters
        ----------
        data : tensorflow SparseTensor
            Data over which to calculate the ELBO.  If not training data,
            must also pass `log_phi`.
        calc_phi : bool, optional
        as_components : bool, optional
            If true, return both the elbo and its individual components.  Useful
            for tracking changes in parameters over iterations.

        Returns
        -------
        ELBO : tensor
            The ELBO
        components : list,
            Contributing components of the ELBO: [xi_contrib, eta_contrib,
            theta_contrib, beta_contrib, z_contrib]. Only returned when
            `as_components` is true.


        """
        with tf.name_scope('ELBO'):
            # convenience
            a = tf.saturate_cast(self.hyper_p.a,self.theta.dtype)
            ap = tf.saturate_cast(self.hyper_p.ap,self.theta.dtype)
            bp = tf.saturate_cast(self.hyper_p.bp,self.theta.dtype)
            c = tf.saturate_cast(self.hyper_p.c,self.beta.dtype)
            cp = tf.saturate_cast(self.hyper_p.cp,self.beta.dtype)
            dp = tf.saturate_cast(self.hyper_p.dp,self.beta.dtype)

            # expectation w.r.t q over sum_cells( log p(xi) ) + entropy
            e_xi_constant = ap * tf.log(bp) - tf.lgamma(ap)
            e_xi_var = (ap-1) * self.xi.e_logx  - bp * self.xi.e_x
            xi_contrib = e_xi_constant + tf.reduce_sum(e_xi_var) \
                    + tf.reduce_sum(self.xi.entropy)

            # expectation w.r.t q over sum_genes( log p(eta) ) + entropy
            e_eta_constant = cp * tf.log(dp) - tf.lgamma(cp)
            e_eta_var = (cp-1) * self.eta.e_logx  - dp * self.eta.e_x
            eta_contrib = e_eta_constant + tf.reduce_sum(e_eta_var) \
                    + tf.reduce_sum(self.eta.entropy)

            # expectation w.r.t q over sum_cells(sum_genes(log p(theta|xi)))
            #   + entropy
            theta_contrib = tf.reduce_sum(a * self.xi.e_logx - tf.lgamma(a) \
                    + ((a-1) * self.theta.e_logx) \
                    - self.xi.e_x * self.theta.e_x + self.theta.entropy)

            # expectation w.r.t q over sum_cells(sum_genes(log p(beta|eta)))
            #   + entropy
            beta_contrib = tf.reduce_sum(c * self.eta.e_logx - tf.lgamma(c) \
                    + ((c-1) * self.beta.e_logx) \
                    - self.eta.e_x * self.beta.e_x + self.beta.entropy)

            # TODO check for dimension match if log_phi is None
            if calc_phi==False and self.z is not None:
                log_phi = self.z.log_phi
            else:
                log_rho = CAVICalculator.get_multinomial_unnorm_log_update(
                        data.indices, self.theta.e_logx, self.beta.e_logx)
                log_phi = tf.subtract(log_rho, tf.reduce_logsumexp(
                    log_rho, axis=1, keep_dims=True))
            e_z = tf.expand_dims(data.values, 1) * tf.exp(log_phi)
            z_const = tf.reduce_sum(-tf.lgamma(data.values+1.0))
            gather_theta_k = lambda x: tf.gather(params=x, indices=data.indices[:,0])
            gather_beta_k = lambda x: tf.gather(params=x, indices=data.indices[:,1])
            e_log_theta_k = tf.transpose( tf.map_fn(gather_theta_k,
                tf.transpose(self.theta.e_logx)) )
            e_log_beta_k = tf.transpose( tf.map_fn(gather_beta_k,
                tf.transpose(self.beta.e_logx)) )
            z_contrib = tf.reduce_sum(e_z * (e_log_theta_k + e_log_beta_k)) \
                    - tf.reduce_sum(self.poisson_rate) + z_const

            elbo = xi_contrib + eta_contrib + theta_contrib + beta_contrib \
                    + z_contrib
            if as_components:
                return elbo, [xi_contrib, eta_contrib, theta_contrib,
                        beta_contrib, z_contrib]
            else:
                return elbo


    def write_params_to_file(self, outdir, prefix='', npy=True,
            save_expectations=False, check_exists=False, save_variational=True):
        """ Write variational parameters and their expectations to file.

        Parameters
        ----------
        outdir : str
            output directory
        prefix : str, optional
            prefix for files
        npy : bool, optional
            If True, save in binary .npy format (default behaviour). Otherwise save
            in space-delimited format
        save_expectations : bool, optional
            If True, save expectations as well as parameters
        check_exists : bool, optional
            If True, check if the output directory exists and add datetime string
            if it does. Default false.
        save_variational : bool , optional [default True]
            Save variational parameters to file

        Note
        ----
        This sort of explicit conversion before writing to disk is not really
        in the spirit of tensorflow/in accordance with best practices, but
        again, it is convenient.
        """
        # check we're saving something
        if not (save_variational or save_expectations):
            msg = 'Either or both of `save_variational` and'
            msg += ' `save_expectations` must be true.  Received'
            msg += ' {} and {}'.format(save_variational, save_expectatons)
            raise ValueError(msg)

        # check if standard named files already exist
        if check_exists:
            try:
                os.makedirs(outdir, exist_ok=False)
            except OSError as e:
                outdir = '{}/{}'.format(outdir,
                        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
                os.makedirs(outdir, exist_ok=False)
        else:
            os.makedirs(outdir, exist_ok=True)

        outstr = '{0}/{1}{2}.{3}'.format(outdir, prefix, \
                '{}', 'npy' if npy else 'txt')
        savefunc = np.save if npy else np.savetxt
        sess = get_session()
        if save_variational:
            self.hyper_p.write_to_file(outdir, prefix)
            savefunc(outstr.format('theta_shape'), sess.run(self.theta.shape))
            savefunc(outstr.format('theta_invrate'), sess.run(
                self.theta.invrate))

            savefunc(outstr.format('beta_shape'), sess.run(self.beta.shape))
            savefunc(outstr.format('beta_invrate'), sess.run(self.beta.invrate))

            try:
                savefunc(outstr.format('xi_shape'), sess.run(self.xi.shape))
            except IndexError:
                with open(outprefix + 'xi_shape.txt', 'w') as f:
                    f.write(str(sess.run(self.xi.shape)))
            savefunc(outstr.format('xi_invrate'), sess.run(self.xi.invrate))

            try:
                savefunc(outstr.format('eta_shape'), sess.run(self.eta.shape))
            except IndexError:
                with open(outprefix + 'eta_shape.txt', 'w') as f:
                    f.write(str(sess.run(self.eta.shape)))
            savefunc(outstr.format('eta_invrate'), sess.run(self.eta.invrate))

            savefunc(outstr.format('log_rho'), sess.run(self.z.log_rho))

        if save_expectations:
            savefunc(outstr.format('theta_e_x'), sess.run(self.theta.e_x))
            savefunc(outstr.format('beta_e_x'), sess.run(self.beta.e_x))
            savefunc(outstr.format('xi_e_x'), sess.run(self.xi.e_x))
            savefunc(outstr.format('eta_e_x'), sess.run(self.eta.e_x))
            try:
                savefunc(outstr.format('log_phi'), sess.run(self.z.log_phi))
            except AttributeError:
                print('Could not save log_phi because auxilliary params not in'
                        + ' VariationalParams instance')


    def write_score_to_file(self, outdir, prefix='', score='all', npy=True):
        """ Write scores to file

        Parameters
        ----------
        outdir : str
            output directory
        prefix : str, optional
            prefix for files
        score : str, optional
            score to write.  valid values are: {`norm`, `hnorm`, `termscore`,
            `all`}. Default `all`.
        npy : bool, optional
            If True, save in binary .npy format (default behaviour). Otherwise save
            in space-delimited format
        """
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        outstr = '{0}/{1}{2}.{3}'.format(outdir, prefix, '{}',
                'npy' if npy else 'txt')
        savefunc = np.save if npy else np.savetxt
        sess = get_session()

        valid = ['norm', 'hnorm', 'termscore', 'all']
        if score not in valid:
            msg = 'Invalid score type {}. '.format(score)
            msg += '  Valid score types are [{0}]'.format(', '.join(valid))
            raise InvalidArgumentException

        if score in ['norm', 'all']:
            savefunc(outstr.format('cell_norm'), sess.run(self.theta.norm))
            savefunc(outstr.format('gene_norm'), sess.run(self.beta.norm))
        if score in ['hnorm', 'all']:
            savefunc(outstr.format('cell_hnorm'), sess.run(self.xinorm))
            savefunc(outstr.format('gene_hnorm'), sess.run(self.etanorm))
        if score in ['termscore', 'all']:
            savefunc(outstr.format('cell_termscore'), sess.run(self.theta.termscore))
            savefunc(outstr.format('gene_termscore'), sess.run(self.beta.termscore))

