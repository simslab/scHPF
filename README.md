# Single-cell Hierarchical Poisson Factorization

## About
scHPF is a tool for _de novo_ discovery of both discrete and continuous expression patterns in single-cell RNA\-sequencing (scRNA-seq). We find that scHPF’s sparse low-dimensional representations, non-negativity, and explicit modeling of variable sparsity across genes and cells produce highly interpretable factors.

Algorithmic details, benchmarking against alternative methods, and scHPF's application to a spatially sampled high-grade glioma can be found in our [paper at Molecular Systems Biology](http://msb.embopress.org/content/15/2/e8557.full.pdf).

## Installation
### Environment & Dependencies
scHPF requires Python >= 3.6 and the packages:
- numba <=0.40, >=0.35 (numba 0.41 reduces scHPF's performance. Currently working to resolve.)
- scikit-learn
- pandas
- (optional) loompy


The easiest way to setup a python environment for scHPF is with [anaconda](https://www.continuum.io/downloads).
```
conda create -n schpf_p37 python=3.7 scikit-learn numba=0.40 pandas

# older anaconda
source activate schpf_p37
# XOR newer anaconda
conda activate schpf_p37

# Optional, for using loom files as input to preprocessing
pip install -U loompy
```
### Installing from source
Once you have set up the environment, clone this repository and install.
```
git clone git@github.com:simslab/scHPF.git
cd scHPF
pip install .
```

## scHPF Command Line Interface (CLI) workflow
### Preprocessing
#### Input file formats
scHPF's prep command intakes a molecular count matrix for an scRNA-seq experiment and formats it for training. Note that scHPF is specifically designed for scRNA-seq data with unique molecular identifiers (UMIs), and only takes integer molecular counts. 

scHPF prep currently accepts two input file formats:
1. A whitespace-delimited matrix formatted like: <pre>ENSEMBL_ID  GENE_NAME  UMICOUNT_CELL0  UMICOUNT_CELL1 ... </pre> The matrix should not have a header, but may be compressed with gzip or bzip2. 

2. A loom file ([loompy.org](http://loompy.org/)). The loom file must have at least one of the row attributes `Accession` or `Gene`, where `Accession` is an ENSEMBL id and `Gene` is a gene name. 

#### Filtering non-coding genes during preprocessing
We recommend restricting analysis to protein-coding genes. The `-w`/`--whitelist` option removes all genes in the input data that are **not in** a two column, tab-delimited text file of ENSEMBL gene ids and names. 

Whitelists for human and mouse are provided in the [resources folder](https://github.com/simslab/scHPF/tree/rewrite_release/resources).

*Filtering txt input details:* For whitespace-delimited UMI-count files, filtering is performed using the input matrix's `ENSEMBL_ID` by default, but can be done with `GENE_NAME` using the `--filter-by-gene-name` flag. This is useful for data that does not include a gene id. 

*Filtering loom input details:* For loom files, we filter the `Accession` row attribute against the whitelist's `ENSEMBLE_ID` if `Accession` is present in the loom's row attributes, and filter the `Gene` row attribute against the `GENE_NAME` in the whitelist otherwise. 

#### Running the prep command
To preprocess genome-wide UMI counts for a typical run, use the command:
```
scHPF prep -i UMICOUNT_MATRIX -o OUTDIR -m 5 -w GENE_WHITELIST
```
As written, the command formats data for training and only includes genes that are:
- on the whitelist (eg protein coding, see [resources folder](https://github.com/simslab/scHPF/tree/rewrite_release/resources)) and 
- that we observe transcripts of in at least 5 cells. 


After running this command, `OUTDIR` should contain a matrix market file, `train.mtx`, and an ordered list of genes, `genes.txt`. An optional prefix argument can be added, which is prepended to to the output file names.

More options and details for preprocessing can be viewed with 
```
scHPF prep -h
```

### Training
#### Input file formats
scHPF's train command accepts two formats:
1. Matrix Market (.mtx) files, where rows are cells, columns are genes, and values are nonzero molecular counts. Matrix market files are output by the current scHPF prep command.
2. Tab-delimited COO matrix coordinates, output by the previous version of the preprocessing command. These files are essentially the same as .mtx files, except they do not have a header and are zero indexed. 

#### Running the train command
To train an scHPF using data output from the prep command:
```
scHPF train -i TRAIN_FILE -o OUTDIR -p PREFIX -k 7 -t 5
```
This command performs approximate Bayesian inference on scHPF with, in this instance, seven factors and five different random initializations. scHPF will automatically select the trial with the highest log-likelihood, and save the model in the `OUTDIR` in a serialized [joblib](https://scikit-learn.org/stable/modules/model_persistence.html) file. 

##### Note
If you get an error like *Inconsistency detected by ld.so: dl-version.c: 224: \_dl_check_map_versions* and are running numba 0.40.0, try downgrading to 0.39.0

More options and details for training can be viewed with 
```
scHPF train -h
```

### Extracting cell scores, gene scores, and ranked gene lists
To get gene- and cell-scores in a tab-delimited file, ordered like the genes and cells in the train file and with a column for each factor:
```
scHPF score -m MODEL_JOBLIB -o OUTDIR -p PREFIX
```
To also generate a tab-delimited file of gene names, ranked by gene-score for each factor:
```
scHPF score -m MODEL_JOBLIB -o OUTDIR -p PREFIX -g GENE_FILE
```
`GENE_FILE` is intended to be the `gene.txt` file output by the scHPF prep command (see above), but can in theory be any tab-delimited file where the number of rows is equal to the number of genes in the scHPF model. The score command automatically uses the 1st (zero-indexed) column of `GENE_FILE` (or the only column if there is only one); however, the column used can be specified with `--name-col`.


## scHPF API workflow
Coming soon. This implementation has a scikit-learn-like interface.


##  Citation

```
@article {msb2019scHPF,
	author = {Levitin, Hanna Mendes and Yuan, Jinzhou and Cheng, Yim Ling and Ruiz, Francisco JR and Bush, Erin C and Bruce, Jeffrey N and Canoll, Peter and Iavarone, Antonio and Lasorella, Anna and Blei, David M and Sims, Peter A},
	title = {De novo gene signature identification from single-cell RNA-seq with hierarchical Poisson factorization},
	volume = {15},
	number = {2},
	elocation-id = {e8557},
	year = {2019},
	doi = {10.15252/msb.20188557},
	publisher = {EMBO Press},
	URL = {http://msb.embopress.org/content/15/2/e8557},
	eprint = {http://msb.embopress.org/content/15/2/e8557.full.pdf},
	journal = {Molecular Systems Biology}
}
```

## Help and support
More detailed tutorials on using scHPF to analyze scRNA-seq data, such as code for enrichment analysis on factors, worked jupyter notebooks, and an example selecting the number of factors will be posted soon. In the meantime please [open an issue](https://github.com/simslab/scHPF/issues/new) and I will try to provide whatever help and guidance I can.

## Contributing
Contributions to scHPF are welcome. Please get in touch if you would like to discuss. To contribute, please [fork scHPF](https://github.com/simslab/scHPF/issues#fork-destination-box), make your changes, and submit a pull request.

