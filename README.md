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

The easiest way to setup an environment is with [anaconda](https://www.anaconda.com/download/#macos)
```
conda create -n schpf_p37 python=3.7 scikit-learn numba pandas

# older anaconda
source activate schpf_p37
# newer anaconda
conda activate schpf_p37

# Optional, for using loom files as input to preprocessing
pip install -U loompy
```

Once you have completed requirements, clone this git reposity and install.
```
git clone URL
cd scHPF
python setup.py install
```

Once I merge into master, you should be able to install directly with pip:
```
pip install git+https://www.github.com/simslab/scHPF.git#egg=scHPF
```

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
