.. _joblib: https://scikit-learn.org/stable/modules/model_persistence.html

.. _train-cli:

**********
Train CLI
**********

Basic usage
===========
To train an scHPF using data output from the |scHPF prep command|_:

.. |scHPF prep command| replace:: ``scHPF prep`` command
.. _scHPF prep command: prep-cli.html

.. code:: bash

    scHPF train -i TRAIN_FILE -o OUTDIR -p PREFIX -k 7 -t 5

This command performs approximate Bayesian inference on scHPF with, in this
instance, seven factors and five different random initializations. scHPF will
automatically select the trial with the highest log-likelihood, and save the
model in the OUTDIR in a serialized `joblib`_ file.

Input file format
=================
scHPF's train command accepts two formats:

1. Matrix Market (.mtx) files, where rows are cells, columns are genes, and
   values are nonzero molecular counts. Matrix market files are output by the
   current |scHPF prep command|_.
2. Tab-delimited COO matrix coordinates, output by a previous version of the
   preprocessing command. These files are essentially the same as .mtx files,
   except they do not have a header and are zero indexed.


Debugging
=========
- If you get an error like "Inconsistency detected by ld.so: dl-version.c: 224:
  _dl_check_map_versions" and are running numba 0.40.0, try downgrading to
  0.39.0.

Complete options
================

.. argparse::
   :filename: ../bin/scHPF
   :func: _parser
   :prog: scHPF
   :path: train
