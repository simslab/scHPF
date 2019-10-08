
.. _score-cli:

***********
scHPF score
***********

Basic usage
===========
To get gene- and cell-scores in a tab-delimited file, ordered like the genes and
cells in the train file and with a column for each factor:

.. code:: bash

    scHPF score -m MODEL_JOBLIB -o OUTDIR -p PREFIX

To also generate a tab-delimited file of gene names, ranked by gene-score for
each factor:

.. code:: bash

    scHPF score -m MODEL_JOBLIB -o OUTDIR -p PREFIX -g GENE_FILE

``GENE_FILE`` is intended to be the gene.txt file output by the 
|scHPF prep command|_, but can in theory be any tab-delimited file where the
number of rows is equal to the number of genes in the scHPF model. The score
command automatically uses the 1st (zero-indexed) column of ``GENE_FILE`` (or
the only column if there is only one); however, the column used can be specified
with ``--name-col``.

.. |scHPF prep command| replace:: ``scHPF prep`` command
.. _scHPF prep command: prep-cli.html

If ``OUTDIR`` is omitted, the command will make a new subdirectory of the
directory containing the model.  The new subdirectory will have the same name as
the model file, but without the joblib extension.

The command also outputs files which can be used to 
:ref:`select the number of factors<select-k>` using trained models.

Complete options
================

For complete options, see the :ref:`complete CLI reference<cli-score>` or use the
``-h`` option on the command line:

.. code:: bash

    scHPF score -h
