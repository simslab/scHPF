.. _install:

************
Installation
************

Environment & Dependencies
==========================

scHPF requires Python >= 3.6 and the packages:

*   numba <=0.40, >=0.35 (numba 0.41 reduces performance.)
*   scikit-learn
*   pandas
*   (optional) loompy


The easiest way to setup a python environment for scHPF is with `anaconda`_ (or its stripped-down version `miniconda`_):

.. _anaconda: https://www.anaconda.com/distribution
.. _minconda: https://www.continuum.io/en/latest/miniconda.html

.. code:: bash

    conda create -n schpf_p37 python=3.7 scikit-learn numba=0.40 pandas

    # older anaconda
    source activate schpf_p37
    # XOR newer anaconda
    conda activate schpf_p37

    # Optional, for using loom files as input to preprocessing
    pip install -U loompy


Installing via Git
==================

Once you have set up the environment, clone ``simslab/scHPF`` from github and install.

.. code:: bash

    git clone git@github.com:simslab/scHPF.git
    cd scHPF
    pip install .
