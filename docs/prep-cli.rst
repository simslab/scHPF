.. _loompy docs: http://loompy.org/
.. _resources folder: https://github.com/simslab/scHPF/tree/rewrite_release/resources

.. _prep-cli:

**********
Prep CLI
**********

scHPF prep
==========

To preprocess genome-wide UMI counts for a typical run, use the command:

.. code:: bash

    scHPF prep -i UMICOUNT_MATRIX -o OUTDIR -m 10 -w GENE_WHITELIST

As written, the command prepares a 
:ref:`matrix of molecular counts <matrix-format>` for training and only includes
genes that are:

- on the :ref:`whitelist<whitelist>` and

- that we observe in at least 10 cells.

After running this command, ``OUTDIR`` should contain a matrix market file, ``train.mtx``, and an ordered list of genes, ``genes.txt``. An optional prefix argument can be added, which is prepended to to the output file names.

More options and details for preprocessing can be viewed with

.. code:: bash

    scHPF prep -h


.. _matrix-format:

Input matrix format
===================
``scHPF prep`` takes a molecular count matrix for an scRNA-seq experiment
and formats it for training.  Note that scHPF is specifically designed for data
with unique molecular identifiers (UMIs) and only accepts integer molecular
counts.

1. A **whitespace-delimited matrix** formatted as follows, with no header::

    ENSEMBL_ID    GENE_NAME    UMICOUNT_CELL0    UMICOUNT_CELL1 ...

2. A **loom file** (see `loompy docs`_).  The loom file must have at least one
   of the row attributes ``Accession`` or ``Gene``, where ``Accession`` is an
   ENSEMBL id and ``Gene`` is a gene name. 

.. _whitelist:

Whitelisting genes
==================

About
-----
We recommend restricting analysis to protein-coding genes. The
``-w``/``--whitelist`` option removes all genes in the input data that are *not
in* a two column, tab-delimited text file of ENSEMBL gene ids and names.
Symmetrically, the ``-b``/``--blacklist`` option removes all genes that are *in*
a file.

Whitelists for human and mouse are provided in the `resources folder`_, and
details on construction/custom lists are in the 
:ref:`gene list documentation <genelists>`.

Whitespace-delimited input
--------------------------
For whitespace-delimited UMI-count files, filtering is performed using the input
matrix's ``ENSEMBL_ID`` by default, but can be done with ``GENE_NAME`` using the
``--filter-by-gene-name`` flag. This is useful for data that does not include a
gene id.


loom input
----------
For loom files, we filter the loom ``Accession`` row attribute against the
whitelist's ``ENSEMBLE_ID`` if ``Accession`` is present in the loom's row
attributes, and filter the loom's ``Gene`` row attribute against the
``GENE_NAME`` in the whitelist otherwise.

