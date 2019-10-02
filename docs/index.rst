.. scHPF documentation master file, created by
   sphinx-quickstart on Mon Jul  8 17:02:06 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Single-cell Hierarchical Poisson Factorization
==============================================

Single-cell Hierarchical Poisson Factorization (scHPF) is a tool for *de novo* discovery 
of discrete and continuous expression patterns in single-cell RNA\-sequencing 
(scRNA-seq). 

We find that scHPF’s sparse low-dimensional representations, non-negativity, 
and explicit modeling of variable sparsity across genes and cells produces highly interpretable 
factors.  The algorithm takes genome-wide molecular counts as input, avoids prior normalization, 
and has fast, memory-efficient inference on sparse scRNA-seq datasets. 

Algorithmic details, benchmarking against alternative methods, and scHPF's 
application to a spatially sampled high-grade glioma can be found in `our paper at Molecular Systems Biology`_.

.. _our paper at Molecular Systems Biology: https://doi.org/10.15252/msb.20188557

You can find the software `on github <https://github.com/simslib/scHPF>`_.

.. toctree::
    :maxdepth: 1
    :caption: Setup 

    install
    genelists

.. toctree::
    :maxdepth: 2
    :caption: Commandline workflow

    prep-cli
    train-cli
    score-cli

.. toctree::
    :maxdepth: 2
    :caption: Advanced options

    select_k
    project


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
