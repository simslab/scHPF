#!/usr/bin/python
import warnings
import yaml
import numpy as np
import tensorflow as tf

from util import get_session

class HyperParams:
    """
    Composite class of hyperparameters, which do not change during inference
    (or only change during emperical bayes)

    For each hyperparameter, stores an initial value (ie bp0) and base value
    for the value before transformation with a function that limits range .
    The hyperparameter's value is a range-limiting function of it's base value
    (ie bp = softplus(bp_base), and bp0=initial values of bp and bp0==bp if values
    not changed during inference).


    The above formulation is unnecessarily complicated for scHPF with empirically
    set but fixed hyperparameters, as reported in the scHPF manuscript, but
    allows for Empirical Bayes with minimal modification to the code if desired.

    """
    @staticmethod
    def  load_from_file(yaml_file, dtype=tf.float64):
        """
        Load hyperparameters from file

        Parameters
        ----------
        yaml_file : str
            YAML file with hyperparameter values.  Must include values for
            a, ap, bp, c, cp, dp, ngenes, ncells, and nfactors.  Other
            values (ie a0 or a_base) optional and will be infered if not given.
        dtype : tensorflow datatype
            datatype to use for Hyperparameters

        Returns
        -------
        hyperparams : HyperParams
            Hyperparams object with values set from file
        """
        with open(yaml_file, 'r') as stream:
            params = yaml.load(stream)
            # required
            a  = params['a']
            ap = params['ap']
            bp = params['bp']
            c  = params['c']
            cp = params['cp']
            dp = params['dp']
            ngenes = params['ngenes']
            ncells = params['ncells']
            K = params['nfactors']

            #optional
            getparam = lambda k : params[k] if k in params.keys() else None
            a0  = getparam('a0')
            ap0 = getparam('ap0')
            bp0 = getparam('bp0')
            c0  = getparam('c0')
            cp0 = getparam('cp0')
            dp0 = getparam('dp0')
            a_base  = getparam('a_base')
            ap_base = getparam('ap_base')
            bp_base = getparam('bp_base')
            c_base  = getparam('c_base')
            cp_base = getparam('cp_base')
            dp_base= getparam('dp_base')


            print(params)
        hp =  HyperParams(nfactors=K, ncells=ncells, ngenes=ngenes, a=a, ap=ap,
                bp=bp, c=c, cp=cp, dp=dp, a0=a0, ap0=ap0, bp0=bp0, c0=c0,
                cp0=cp0, dp0=dp0, a_base=a_base, ap_base=ap_base, bp_base=bp_base,
                c_base=c_base, cp_base=cp_base, dp_base=dp_base, dtype=dtype)
        return hp


    def __init__(self, nfactors, ncells, ngenes, a0=None, ap0=None,
            bp0=None, c0=None, cp0=None, dp0=None, a_base=None, ap_base=None,
            bp_base=None, c_base=None, cp_base=None, dp_base=None,
            a=0.3, ap=1, bp=1, c=0.3, cp=1, dp=1, dtype=tf.float64):
        """Initialize hyperparameters

        Parameters
        ----------
        nfactors : int
            number of factors in model
        ncells : int
            number of cells
        ngenes : int
            number of genes
        a : float, default 0.3
            shape hyperparameter for cell factor weights
        ap : float, default 1
            shape hyperparameter for cell budgets
        bp: float, default 1
            inverse rate hyperparam for cell budgets.
            default value not recommended
        c : float, default 0.3
            shape hyperparameter for gene factor weights
        cp : float, default 1
            shape hyperparameter for gene budgets
        dp: float, default 1
            inverse rate hyperparam for gene budgets.
            default value not recommended
        a0 through dp0: floats, default None
            Target values for hyperparameters (inverse transformed to get _base
            values if _base values not given)
        a_base through dp_base: floats
            base values passed through range-limiting functions to calculate

        """
        # tf initializers for inverses of activation-like functions that
        # constrain the ranges of the hyperparameters
        def invsoftplus(y):
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    return tf.constant(np.log(np.exp(y) - 1), dtype=dtype)
                except Warning as w:
                    return tf.constant(y, dtype=dtype)

        def invsigmoid(y):
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    return tf.constant(-np.log( 1/y - 1), dtype=dtype)
                except Warning as w:
                    return tf.constant(y, dtype=dtype)

        # hyperparameters governing model
        self.nfactors = nfactors     # K
        self.ncells = ncells         # M
        self.ngenes = ngenes         # N

        with tf.variable_scope('hyperparams'):
            # sigmoid
            with tf.name_scope('a'):
                self.a0 = a if a0 is None else a0
                a_base_init = invsigmoid(self.a0) if a_base is None else a_base
                self.a_base = tf.get_variable('a_base', initializer=a_base_init)
                self.a = tf.sigmoid(self.a_base)

            # softplus
            with tf.name_scope('ap'):
                self.ap0 = ap if ap0 is None else ap0
                ap_base_init = invsoftplus(self.ap0) if ap_base is None \
                        else ap_base
                self.ap_base = tf.get_variable('ap_base', initializer=ap_base_init)
                self.ap = tf.nn.softplus(self.ap_base)

            # softplus
            with tf.name_scope('bp'):
                self.bp0 = bp if bp0 is None else bp0
                bp_base_init = invsoftplus(self.bp0) if bp_base is None else bp_base
                self.bp_base = tf.get_variable('bp_base', initializer=bp_base_init)
                self.bp = tf.nn.softplus(self.bp_base)

            # sigmoid
            with tf.name_scope('c'):
                self.c0 = c if c0 is None else c0
                c_base_init = invsigmoid(self.c0) if c_base is None else c_base
                self.c_base = tf.get_variable('c_base', initializer=c_base_init)
                self.c = tf.sigmoid(self.c_base)

            # softplus
            with tf.name_scope('cp'):
                self.cp0 = cp if cp0 is None else cp0
                cp_base_init = invsoftplus(self.cp0) if cp_base is None else cp_base
                self.cp_base = tf.get_variable('cp_base', initializer=cp_base_init)
                self.cp = tf.nn.softplus(self.cp_base)

            # softplus
            with tf.name_scope('dp'):
                self.dp0 = dp if dp0 is None else dp0
                dp_base_init = invsoftplus(self.dp0) if dp_base is None else dp_base
                self.dp_base = tf.get_variable('dp_base', initializer=dp_base_init)
                self.dp = tf.nn.softplus(self.dp_base)

            self.dense_shape = (self.ncells, self.ngenes)
            # dtype
            self.dtype = dtype


    def write_to_file(self, outdir, prefix=''):
        """Write values to file

        Parameters
        ----------
        outdir : str
            The output directory for hyperparameters
        perfix : str, default ''
            Prefix for hyperparameters file
        """
        param_dict = self.to_dict()
        outfile = "{0}/{1}hyperparams.yaml".format(outdir,
                prefix.rstrip('.') + '.' if len(prefix)>0 else '')
        with open(outfile, 'w') as f:
            yaml.dump(param_dict, f, default_flow_style=False,
                    allow_unicode=True)


    def to_dict(self):
        """Get  attributes as a dictionary, with tensorflow variables
        evaluated.  May be some minor loss of precision, as may convert
        from float64 to standard 32 bit float

        Returns
        -------
        param_dict : dict
            Dictionary of hyperparameters (evaluated values)
        """
        sess = get_session()
        a, ap, bp, c, cp, dp = sess.run([self.a, self.ap, self.bp, self.c,
            self.cp, self.dp])
        a_base, ap_base, bp_base, c_base, cp_base, dp_base = sess.run(
                [self.a_base, self.ap_base, self.bp_base, self.c_base,
                    self.cp_base, self.dp_base])
        param_dict = {
                'a':float(a), 'ap':float(ap), 'bp':float(bp),
                'c':float(c), 'cp':float(cp), 'dp':float(dp),

                'a_base':float(a_base), 'ap_base':float(ap_base),
                'bp_base':float(bp_base), 'c_base':float(c_base),
                'cp_base':float(cp_base), 'dp_base':float(dp_base),

                'a0':float(self.a0), 'ap0':float(self.ap0),
                'bp0':float(self.bp0), 'c0':float(self.c0),
                'cp0':float(self.cp0), 'dp0':float(self.dp0),

                'nfactors' : self.nfactors,
                'ncells' : int(self.ncells),
                'ngenes' : int(self.ngenes)
                }
        return param_dict
