#!/usr/bin/python

import numpy as np
import tensorflow as tf

def get_session():
  """ Get global session, creating one if it does not exist

  Returns
  -------
  _SCHPF_SESSION : tf.InteractiveSession

  Notes
  -----
  Adapted from Edward: github.com/blei-lab/edward
  """
  global _SC_SESSION
  if tf.get_default_session() is None:
    _SC_SESSION = tf.InteractiveSession()
  else:
    _SC_SESSION = tf.get_default_session()

  return _SC_SESSION


def tf_log_b(t, b=2):
    """ log base b
    Parameters
    ----------
    t : tensor
        tensor to take the log base b of
    b : float
        base for logarithm. Default 2
    Returns
    -------
    log_b : tensor
        log base b or the tensor
    """
    return tf.divide(tf.log(t), tf.log(2), name='log{}'.format(b))


def cv(x, axes, name='cv'):
    """ Compute the coefficient of variaton over the specified axes

    Parameters
    ----------
    x : tensor
    axes : Array of ints or int
        Axes along wich to compute the cv
    name : str
        name to use to scope operations

    Returns
    -------
    cv : tensor
    """
    with tf.name_scope(name):
        m, v = tf.nn.moments(x, axes=axes)
        cv = tf.divide(tf.sqrt(v), m, name=name)
        return tf.where(tf.is_nan(cv), tf.zeros_like(cv), cv)


def poisson_entropy_truncated(rate, truncate_at=50, step=1):
    """Calculate the entropy of a poisson with the given rate,
    truncating the series
    Parameters
    ----------
    rate : tensor
        The rate parameter of the poisson distribution
    truncate_at : int, optional
        Where to truncate that summation term [default 50]
    step : int, optional
        Step size for the series [default 1]

    Returns
    -------
    H : tensor
        The approximate (truncated) entropy of a Poisson with the given rate
    """
    with tf.name_scope('poisson_entropy_truncated'):
        def sum_term(k):
            my_k = tf.saturate_cast(k, rate.dtype)
            loggamma = tf.lgamma(k + tf.constant(1, dtype=rate.dtype))
            num = tf.pow(rate, k) * loggamma
            denom = tf.exp(loggamma)
            return tf.divide(num, denom)

        exact = tf.multiply(rate,tf.constant(1,dtype=rate.dtype) - tf.log(rate))
        multiplier = tf.exp(-rate)
        truncated_sum = tf.reduce_sum( tf.map_fn(
            sum_term, tf.cast(tf.range(0,truncate_at,step),rate.dtype)), axis=0)
        truncated_H = exact + multiplier * truncated_sum
    return truncated_H


def poisson_entropy_large_rate(rate):
    """Calculate the entropy of a poisson with the given rate using the
    approximation for large rates

    Parameters
    ----------
    rate : tensor
        The rate parameter of the poisson distribution

    Returns
    -------
    H : tensor
        The approximate entropy of a Poisson with the given (large) rate
    """
    with tf.name_scope('poisson_entropy_large_rate'):
        pi = tf.constant(np.pi, dtype=rate.dtype)
        e = tf.constant(np.e, dtype=rate.dtype)
        term0 = tf.constant(0.5, dtype=rate.dtype) * tf.log(2*pi*e*rate)
        term1 = -tf.divide(tf.constant(1/12, dtype=rate.dtype), rate)
        term2 = -tf.divide(tf.constant(1/24, dtype=rate.dtype), tf.pow(rate, 2))
        term3 = -tf.divide(tf.constant(19/360, dtype=rate.dtype), tf.pow(rate, 3))
        term4 = tf.divide(tf.ones((), dtype=rate.dtype), tf.pow(rate, 4))
        result = term0 + term1 + term2 + term3 + term4
        return result


def poisson_entropy_approx(rate, truncate_at=50, step=1):
    """Calculate the entropy of a Poisson using a truncated sum, and the large
    rate approixation when truncation fails

    Parameters
    ----------
    rate : tensor
        The rate parameter of the Poisson.  Must > 0 for all values.
    truncate_at : int, optional
        Where to truncate that summation term [default 50]
    step : int, optional
        Step size for the series [default 1]

    Returns
    -------
    H : tensor
        approximate entropy of the Poisson
    """
    with tf.name_scope('poisson_entropy_approx'):
        pois_trunc = poisson_entropy_truncated(rate, truncate_at, step)
        pois_large = poisson_entropy_large_rate(rate)

        use_large = tf.logical_or(tf.is_nan(pois_trunc),
                tf.less_equal(pois_trunc, tf.zeros_like(rate)))

        return tf.where(use_large, pois_large, pois_trunc,
                name='poisson_entropy_approx')


def log_likelihood_poisson(poisson_rate, data, name='llh_pointwise'):
    """
    Log likelihood for data given a poisson rate

    poisson_rate : the poisson mean/variance
    data : the observed data
    """
    with tf.name_scope(name):
        return data * tf.log(poisson_rate) - poisson_rate - \
                tf.lgamma(data+tf.constant(1.0, dtype=data.dtype))


def create_sparse_tensor(indices, values, ncells, ngenes, dtype=tf.float64):
    """
    Parameters
    ----------
    indices : numpy array
        An (nsamples, 2) shaped numpy array of integers
    values :numpy array
        An (nsamples, ) shaped numpy array
    ncells : int
        Number of cells (rows)
    ngenes : int
        Number of genes (columns)
    dtype :
        tensorflow datatype for values
    """
    new_ix = indices[:,:2]
    values = tf.cast(values, dtype)
    try :
        return tf.SparseTensor(indices=new_ix, values=values,
                shape=(ncells,ngenes))
    except TypeError:
        return tf.SparseTensor(indices=new_ix, values=values,
                dense_shape=(ncells,ngenes))

def get_median_1d(v):
    with tf.name_scope('median'):
        v = tf.reshape(v, [-1])
        m = tf.shape(v)[0]//2
        return tf.nn.top_k(v, m).values[m-1]
