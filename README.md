# Single-cell Hierarchical Poisson Factorization

Pre-release of [Single-cell Hierarchical Poisson Factorization (scHPF)](https://www.biorxiv.org/content/early/2018/07/11/367003), as described in the forthcoming manuscript: <br/> *De novo* Gene Signature Identification from Single-Cell RNA-Seq with Hierarchical Poisson Factorization.

scHPF is a tool for _de novo_ discovery of both discrete and continuous expression patterns in single-cell RNA\-sequencing (scRNA-seq) data. It adapts [Hierarchical Poisson Factorization](http://www.cs.columbia.edu/~blei/papers/GopalanHofmanBlei2015.pdf) to avoid prior normalization and model variable sparsity across genes and cells. Algorithmic details, benchmarking against alternative methods, and scHPF's application to a spatially sampled high-grade glioma can be found in our [paper on biorXiv](https://www.biorxiv.org/content/early/2018/07/11/367003).

## Requirements
Code for preprocessing, training, and postprocessing has been tested with Python 3.6 and Tensorflow 1.3/1.8 on Ubuntu and Mac.

scHPF requires the Python packages:
- numpy
- pandas
- pyyaml
- tensorflow (for CPU)
- (optional) seaborn

For easy startup, tensorflow can also be [installed from a prebuilt binary](https://www.tensorflow.org/install/). For example, the environment can be set up with anaconda on Mac as follows:
```
conda create -n tensorflow_p36 pip python=3.6 numpy scipy pandas pyyaml seaborn cython
source activate tensorflow_p36
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.8.0-py3-none-any.whl
```
Binaries for other operating systems are available in the [tensorflow installation guide](https://www.tensorflow.org/install/).  Installing [tensorflow from source](https://www.tensorflow.org/install/install_source) may increase computational performance.  Regardless of how you install, we **strongly** recommend [validating your installation](https://www.tensorflow.org/install/install_linux#run_a_short_tensorflow_program) before proceeding.

Once you have completed requirements, clone this git reposity.

## scHPF workflow
### Preprocessing
scHPF's preprocessing.py command intakes a molecular count matrix for an scRNA-seq experiment with unique molecular identifiers (UMIs).  The whitespace-delimited matrix should be formatted like:
> <pre>ENSEMBL_ID  GENE_SYMBOL  UMICOUNT_CELL0  UMICOUNT_CELL1 ... </pre>

The matrix should not have a header, but may be compressed with gzip or bzip2. We note that scHPF is specifically designed for scRNA-seq data with UMIs, and only takes integer molecular counts.

To preprocess the whitespace-delimited count matrix for a typical run, use the command:
```
python -m scHPF.preprocessing --input UMICOUNT_MATRIX --prefix PREFIX -o OUTPUT_DIR -m 5 --whitelist GENE_WHITELIST
```

Where OUTPUT\_DIR does not need to exist and GENE\_WHITELIST is a two column, whitespace-delimited text file of ENSEMBL\_IDs and GENE\_SYMBOLs (see resources folder for an example).  As written, the command formats data for training and only includes genes that are (1) on the whitelist (eg protein coding) and (2) that we observe transcripts of in at least 5 cells.  After running this command, OUTPUT\_DIR should contain a matrix 'PREFIX.matrix.txt', a list of genes 'PREFIX.genes.txt', a preprocessing log file 'preprocessing.log.yaml', and a sparse-formatted training data file 'train.tsv'. More options and details for preprocessing can be viewed with 
```
python -m scHPF.preprocessing -h
```

### Training
Both computation time and memory requirements for scHPF scale with the number of non-zero values in the dataset and the number of factors in the model.  For a typical 3' scRNA-seq dataset with UMIs, 90-95% sparsity, and up to 3,000 or 4,000 cells, the model can reasonably be trained on a laptop with a dual-core i7 processor and 16GB RAM (it may run reasonably at lower specs, but we haven't tested it).  For larger numbers of cells, faster computation, or running multiple training sessions at once, we recommend a remote instance such as an AWS EC2 m4.xlarge, m4.2xlarge, or m4.4xlarge instance.  Datasets with more than 15,000 cells and a large number of factors may benefit from memory-intensive instance types and the training flag `--low-mem`, which avoids some memory-intensive summary statistic computations. 

Inference can be run using the train.py program.  For example:
```
python -m scHPF.train -i PREPROCESSING_OUTPUT_DIR -o TRAINING_OUTPUT_DIR -k 7 -t 5
```
This command performs approximate Bayesian inference on scHPF with, in this instance, seven factors and five different random initializations (in sequence). scHPF will automatically select the trial with the highest log likelikhood, and save the model in a subdirectory of TRAINING\_OUTPUT\_DIR with a name that states the value of k used and the id of the best run.  For example, the output of the previous command might be saved in 'TRAINING\_OUTPUT\_DIR/k007.run3;. More options for training can be viewed with
```
python -m scHPF.train -h
```

### Extracting scores
To extract gene and cell scores from a trained scHPF model, run
```
python -m scHPF.postprocessing score --param-dir PARAM_DIR
```
Where PARAM\_DIR is the subdirectory of TRAINING\_OUTPUT\_DIR in which the trained model was saved by the train.py command.  Cell and gene scores will be saved in 'TRAINING\_OUTPUT\_DIR/score/cell\_hnorm.txt' and 'TRAINING\_OUTPUT\_DIR/score/gene\_hnorm.txt'. 'cell\_hnorm.txt' is a cell by factor matrix of cell scores, where cells are in the same order as the original input matrix. 'gene\_hnorm.txt' is a gene by factor matrix of gene scores, where genes are in the same order as the genes in the PREFIX.genes.txt file produced by the preprocessing command.

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
