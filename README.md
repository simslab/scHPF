# Single-cell Hierarchical Poisson Factorization

Pre-release of [Single-cell Hierarchical Poisson Factorization (scHPF)](https://www.biorxiv.org/content/early/2018/07/11/367003), as described in the forthcoming manuscript: <br/> *De novo* Gene Signature Identification from Single-Cell RNA-Seq with Hierarchical Poisson Factorization.

scHPF is a tool for _de novo_ discovery of both discrete and continuous expression patterns in single-cell RNA\-sequencing (scRNA-seq) data. It adapts [Hierarchical Poisson Factorization](http://www.cs.columbia.edu/~blei/papers/GopalanHofmanBlei2015.pdf) to avoid prior normalization and model variable sparsity across genes and cells. Algorithmic details, benchmarking against alternative methods, and scHPF's application to a spatially sampled high-grade glioma can be found in our [paper on biorXiv](https://www.biorxiv.org/content/early/2018/07/11/367003).

## Updates

scHPF has a new, improved implementaiton in numba that includes both a command line interface and a scikit-learn-like API.  It is substailly faster and more memory-efficient than tnesorflow scHPF, espcially when many virtual CPUs are available in a high performance compute cluster or compute service like AWS.  Numba scHPF not currently back-compatible with trained models from tensorflow scHPF, but I will be fixing this very soon.

# Documentation

## Installation

scHPF requires the Python >= 3.6 and the packages:
- numba
- scikit-learn
- pandas
- (optional) loompy

The easiest way to setup a python environment scHPF is with [anaconda](https://www.anaconda.com/download/#macos):
```
conda create -n schpf_p37 python=3.7 scikit-learn numba pandas

# older anaconda
source activate schpf_p37
# newer anaconda
conda activate schpf_p37

# Optional, for using loom files as input to preprocessing
pip install -U loompy
```

Once you have set up the environment, clone this git reposity and install.
```
git clone URL
cd scHPF
python setup.py install
```

## scHPF CLI workflow
### Preprocessing
scHPF's preprocessing.py command intakes a molecular count matrix for an scRNA-seq experiment with unique molecular identifiers (UMIs).  The are currently two options for input file formats:

1. A whitespace-delimited matrix should be formatted like:
> <pre>ENSEMBL_ID  GENE_SYMBOL  UMICOUNT_CELL0  UMICOUNT_CELL1 ... </pre>
The matrix should not have a header, but may be compressed with gzip or bzip2. We note that scHPF is specifically designed for scRNA-seq data with UMIs, and only takes integer molecular counts.

2. A loom file (see [loompy.org](http://loompy.org/)). For filtering against a whitelist or blacklist of genes (recommended to select protein coding genes only), the loom file must have a row attribute 'Gene'.

To preprocess genome-wide UMI counts for a typica run, use the command:
```
scHPF prep --input UMICOUNT_MATRIX --prefix PREFIX -o OUTPUT_DIR -m 0.01 --whitelist GENE_WHITELIST
```

Where OUTPUT\_DIR does not need to exist and GENE\_WHITELIST is a two column, whitespace-delimited text file of ENSEMBL\_IDs and GENE\_SYMBOLs (see resources folder for an example).  As written, the command formats data for training and only includes genes that are (1) on the whitelist (eg protein coding) and (2) that we observe transcripts of in at least 0.1% of cells.  After running this command, OUTPUT\_DIR should contain a matrix 'PREFIX.train.mtx' and a list of genes 'PREFIX.genes.txt'. More options and details for preprocessing can be viewed with 
```
scHPF prep -h
```

### Training
TODO

### Extracting cell scores, gene scores, and ranked gene lists
TODO

##  Citation

```
@article {biorxiv2018scHPF,
    author = {Levitin, Hanna Mendes and Yuan, Jinzhou and Cheng, Yim Ling and Ruiz, Francisco J.R. and Bush, Erin C. and Bruce, Jeffrey N. and Canoll, Peter and Iavarone, Antonio and Lasorella, Anna and Blei, David M. and Sims, Peter A.},
    title = {De novo Gene Signature Identification from Single-Cell RNA-Seq with Hierarchical Poisson Factorization},
    year = {2018},
    doi = {10.1101/367003},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2018/07/11/367003},
    eprint = {https://www.biorxiv.org/content/early/2018/07/11/367003.full.pdf},
    journal = {bioRxiv}
}
```

## Help and support
More detailed tutorials on using scHPF to analyze scRNA-seq data, such as code for enrichment analysis on factors, worked jupyter notebooks, and an example selecting the number of factors will be posted soon. In the meantime please [open an issue](https://github.com/simslab/scHPF/issues/new) and I will try to provide whatever help and guidance I can.

## Contributing
Contributions to scHPF are welcome. Please get in touch if you would like to discuss. To contribute, please [fork scHPF](https://github.com/simslab/scHPF/issues#fork-destination-box), make your changes, and submit a pull request.
