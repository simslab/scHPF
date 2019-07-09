.. _premade lists: https://github.com/simslab/scHPF/tree/master/resources
.. _stable identifiers: https://useast.ensembl.org/info/genome/stable_ids/index.html
.. _biotypes: https://www.gencodegenes.org/pages/biotypes.html

**********
Gene lists
**********

About
=====
We recommend restricting analysis to protein-coding genes, and bundle
`premade lists`_ of coding genes for human and mouse with the scHPF code.  The 
:ref:`prep CLI command <prep-cli>` optionally uses these lists to filter input 
data.  Although ENSEMBL ids are theoretically unambiguous and consistent across
releases (ie `stable identifiers`_), you may want to generate your own list from 
a different annotation or with different parameters for gene inclusion. 

Premade lists
=============
The scHPF code includes tab-delimited lists of ENSEMBL ids and names for genes 
with protein coding, T-cell receptor constant, or immunoglobulin constant 
`biotypes`_ for human and mouse.

Premade lists can be found in the 
`code's resources folder <https://github.com/simslab/scHPF/tree/master/resources>`_:

    * Human (GENCODE v24, v29, v31)
    * Mouse (GENCODE vM10, vM19)

Format
======
Example tab-delimited gene list::

    ENSG00000186092	OR4F5
    ENSG00000284733	OR4F29
    ENSG00000284662	OR4F16
    ENSG00000187634	SAMD11
    ENSG00000188976	NOC2L
    ENSG00000187961	KLHL17

By default, the prep command assumes a two-column, tab-delimited text file of 
ENSEMBL gene ids and names, and uses the first column (assumed to be ENSEMBL id) 
to filter genes. See the  
:ref:`prep command documentation <prep-cli>` for other options. 

.. note::
    ENSEMBL ids may end in a period followed by an unstable version 
    number (eg ENSG00000186092.6). By default, the prep command ignores anything 
    after the period. This means ``[ENS-ID].[VERSION]`` is equivalent to 
    ``[ENS-ID]`` . See the :ref:`prep command <prep-cli>` for other options. 


Making custom gene lists
========================
Although ENSEMBL ids aim to be unambiguous and consistent across
releases (ie `stable identifiers`_), you may want to generate your own list from 
a different annotation or with different parameters for gene inclusion.


Example creation script
~~~~~~~~~~~~~~~~~~~~~~~
Reference files of ids and names for genes with with 
``protein_coding``, ``TR_C_gene``, or ``IG_C_gene`` biotypes in the GENCODE 
main annotation (in this case ``gencode.v29.annotation.gtf``) were generated as follows:

.. code:: bash

    # Select genes with feature gene and level 1 or 2
    awk '{if($3=="gene" && $0~"level (1|2);"){print $0}}' gencode.v29.annotation.gtf > gencode.v29.annotation.gene_l1l2.gtf 

    # Only include biotypes protein_coding, TR_C_g* and IG_C_g*
    awk '{if($12~"TR_C_g" || $12~"IG_C_g" || $12~"protein_coding"){print $0}}' gencode.v29.annotation.gene_l1l2.gtf > gencode.v29.annotation.gene_l1l2.pc_TRC_IGC.gtf

    # Retrieve ENSEMBL gene id and name
    awk '{{OFS="\t"}{gsub(/"/, "", $10); gsub(/;/, "", $10); gsub(/"/, "", $14); gsub(/;/, "", $14); print $10, $14}}' gencode.v29.annotation.gene_l1l2.pc_TRC_IGC.gtf > gencode.v29.annotation.gene_l1l2.pc_TRC_IGC.stripped.txt



