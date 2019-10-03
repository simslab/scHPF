.. _install:

************
Installation
************

Environment & Dependencies
==========================

scHPF requires Python >= 3.6 and the packages:

*   numba (:ref:`version requirement depends on python version<numba>`, but will be safe with 0.45)
*   scikit-learn
*   pandas
*   (optional) loompy

The easiest way to setup a python environment for scHPF is with `anaconda`_ (or
its stripped-down version `miniconda`_):

.. _anaconda: https://www.anaconda.com/distribution
.. _miniconda: https://docs.conda.io/en/latest/miniconda.html

.. code:: bash

    conda create -n schpf_p37 python=3.7 scikit-learn numba=0.45 pandas

    # older anaconda
    source activate schpf_p37
    # XOR newer anaconda
    conda activate schpf_p37

    # Optional, for using loom files as input to preprocessing
    pip install -U loompy


.. _numba:

numba compatibility
-------------------
Certain versions (including patch versions) of Python and numba do not play
well together (at least in the context of scHPF), resulting in segmentation
faults and/or horrible performance.  In our experience, combos that
avoid these issues seem to be:

**Python <=3.7.3**
    numba versions: 0.39, 0.40, 0.44, 0.45
**Python 3.7.4**
    numba versions: 0.44, 0.45


Installing scHPF 
================

Once you have set up the environment, clone ``simslab/scHPF`` from github and install.

.. code:: bash

    git clone git@github.com:simslab/scHPF.git
    cd scHPF
    pip install .
