# Single-cell Hierarchical Poisson Factorization

Pre-release of [Single-cell Hierarchical Poisson Factorization (scHPF)](https://www.biorxiv.org/content/early/2018/07/11/367003), as described in the forthcoming manuscript: <br/> *De novo* Gene Signature Identification from Single-Cell RNA-Seq with Hierarchical Poisson Factorization.

## Updates

scHPF has a new, improved implementaiton in numba that includes both a command line interface and a scikit-learn-like API.  It is substailly faster and more memory-efficient than Tensorflow scHPF, espcially when many virtual CPUs are available in a high performance compute cluster or with on depand computing like AWS.  Numba scHPF is not currently back-compatible with trained models from Tensorflow scHPF, but I will be fixing this very soon.

## About
scHPF is a tool for _de novo_ discovery of both discrete and continuous expression patterns in single-cell RNA\-sequencing (scRNA-seq) data. It adapts [Hierarchical Poisson Factorization](http://www.cs.columbia.edu/~blei/papers/GopalanHofmanBlei2015.pdf) to avoid prior normalization and model variable sparsity across genes and cells. Algorithmic details, benchmarking against alternative methods, and scHPF's application to a spatially sampled high-grade glioma can be found in our [paper on biorXiv](https://www.biorxiv.org/content/early/2018/07/11/367003).

# Documentation
## Installation
scHPF requires the Python >= 3.6 and the packages:
- numba
- scikit-learn
- pandas
- (optional) loompy

The easiest way to setup a python environment for scHPF is with [anaconda](https://www.anaconda.com/download/#macos):
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

## scHPF Command Line Interface workflow
### Preprocessing
#### Input file formats
scHPF's preprocessing.py command intakes a molecular count matrix for an scRNA-seq experiment.  We note that scHPF is specifically designed for scRNA-seq data with unique molecular identifiers (UMIs), and only takes integer molecular counts. 

scHPF prep currently accepts two input file formats:
1. A whitespace-delimited matrix formatted like: <pre>ENSEMBL_ID  GENE_SYMBOL  UMICOUNT_CELL0  UMICOUNT_CELL1 ... </pre> The matrix should not have a header, but may be compressed with gzip or bzip2. 

2. A loom file ([loompy.org](http://loompy.org/)). For filtering against a whitelist or blacklist of genes (recommended), the loom file must have a row attribute 'Gene'. Input must have the '.loom' extension.

#### Running the prep command
To preprocess genome-wide UMI counts for a typical run, use the command:
```
scHPF prep -i UMICOUNT_MATRIX -o OUTPUT_DIR -m 0.01 -w GENE_WHITELIST
```
Where OUTPUT\_DIR does not need to exist and GENE\_WHITELIST is a two column, tab-delimited text file of ENSEMBL\_IDs and GENE_SYMBOLs (see resources folder for an example).  As written, the command formats data for training and only includes genes that are:
- on the whitelist (eg protein coding) and 
- that we observe transcripts of in at least 0.1% of cells. 

For whitespace-delimited UMI-count files, filtering is performed using the ENSEMBL_ID (0th column in both UMICOUNT_MATRIX and GENE_WHITELIST) by default, but can be done with gene symbols (1st zero-indexed column) using the --filter-by-gene-name flag.  For loom files, 'Gene' row attibute is used from the input matrix, and filterd against GENE_SYMBOL in the whitlist. 

After running this command, OUTPUT_DIR should contain a matrix `train.mtx` and a list of genes `genes.txt`. An optional prefix argument can be added, which is prepended to to the output file names.

More options and details for preprocessing can be viewed with 
```
scHPF prep -h
```

### Training
#### Input file formats
scHPF's train command accepts two formats:
1. Matrix Market (.mtx) files, where rows are cells, columns are genes, and values are nonzero molecular counts.  Matrix market files are output by the current scHPF prep command.
2. Tab-delimited COO matrix coordinates, output by the previous version of the preprocessing command.  These files are essentially the same as .mtx files, except they do not have a header and are zero indexed. 

#### Running the train command
To train an scHPF using data output from the prep command:
```
scHPF train -i TRAIN_FILE -o OUTPUT_DIR -p PREFIX -k 7 -t 5
```
This command performs approximate Bayesian inference on scHPF with, in this instance, seven factors and five different random initializations (in sequence). scHPF will automatically select the trial with the highest log likelikhood, and save the model in the OUTPUT_DIR in a seralized [joblib](https://scikit-learn.org/stable/modules/model_persistence.html) file. 

More options and details for training can be viewed with 
```
scHPF train -h
```

### Extracting cell scores, gene scores, and ranked gene lists
```
schpf score -m MODEL_JOBLIB -o OUTDIR -p PREFIX
```

To also generate a tab-delimited file of gene names, ranked by gene-score for each factor:
```
schpf score -m MODEL_JOBLIB -o OUTDIR -p PREFIX -g GENE_FILE
```
GENE_FILE is intended to be the `gene.txt` file output by the prep command, but can in theory be any tab-delimited file where the number of rows is equal to the number of genes in the scHPF model. The score command automatically uses the 1st (zero-indexed) column of GENE_FILE (or the only column if there is only one); however, the column used can be specified with --name-col.

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
