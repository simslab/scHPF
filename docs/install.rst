.. _install:

************
Installation
************

Environment & Dependencies
==========================

scHPF requires Python >= 3.6 and the packages:

*   numba (:ref:`version requirement depends on python version<numba>`, but will be safe with 0.45, and probably 0.45+)
*   scikit-learn
*   pandas
*   (optional) loompy

The easiest way to setup a python environment for scHPF is with `anaconda`_ (or
its stripped-down version `miniconda`_):

.. _anaconda: https://www.anaconda.com/distribution
.. _miniconda: https://docs.conda.io/en/latest/miniconda.html

.. code:: bash

    conda create -n schpf_p37 python=3.7 scikit-learn numba=0.45 pandas

    # for newer anaconda versions
    conda activate schpf_p37
    # XOR older anaconda verstions
    source activate schpf_p37

    # Optional, for using loom files as input to preprocessing
    pip install -U loompy


.. _numba:

numba compatibility
-------------------
Certain micro-versions of Python and numba do not play well together, resulting
in segmentation faults and/or horrible performance (at least for the ops scHPF
uses).  In our experience, micro-version combos that avoid these issues are
listed below, as well as known-bad combination, but note this is not an
exhaustive list:

**Python 3.7.9**
    Compatible numba:  0.45-0.50
    DO NOT USE: 0.44 or earlier
**Python 3.7.5 - 3.7.8**
    Not tested
**Python 3.7.4**
    Compatible numba: 0.44, 0.45
    DO NOT USE: 0.43 or earlier
**Python <=3.7.3**
    Compatible numba: 0.39, 0.40, 0.44, 0.45
    DO NOT USE: 0.41-0.43

*Please* let us know about any weird errors/slowness your experience so we can 
document!

Installing scHPF 
================

Once you have set up the environment, clone ``simslab/scHPF`` from github and
install.

.. code:: bash

    git clone git@github.com:simslab/scHPF.git
    cd scHPF
    pip install .
