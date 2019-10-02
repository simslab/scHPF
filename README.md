# Single-cell Hierarchical Poisson Factorization

## About
scHPF is a tool for _de novo_ discovery of both discrete and continuous expression patterns in single-cell RNA\-sequencing (scRNA-seq). We find that scHPF’s sparse low-dimensional representations, non-negativity, and explicit modeling of variable sparsity across genes and cells produce highly interpretable factors.

Algorithmic details, benchmarking against alternative methods, and scHPF's application to a spatially sampled high-grade glioma can be found in our [paper at Molecular Systems Biology](http://msb.embopress.org/content/15/2/e8557.full.pdf).

##  Installation
### Environment & Dependencies
scHPF requires Python >= 3.6 and the packages:
- numba <=0.40, >=0.35 (numba 0.41 reduces scHPF's performance...on TODO list to resolve)
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

## scHPF Command Line Interface (CLI) Quickstart

1. [Prepare your data](https://schpf.readthedocs.io/en/latest/prep-cli.html). 

2. [Train a model](https://schpf.readthedocs.io/en/latest/train-cli.html).

3. [Get gene and cell scores with the CLI](https://schpf.readthedocs.io/en/latest/score-cli.html)


## scHPF API

scHPF has a scikit-learn like API. Trained models are stored in a serialized
joblib format.


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
Please [open an issue](https://github.com/simslab/scHPF/issues/new) and I will try to provide whatever help and guidance I can.

## Contributing
Contributions to scHPF are welcome. Please get in touch if you would like to
discuss/check it's something I've already done but haven't pushed to master yet.
To contribute, please [fork
scHPF](https://github.com/simslab/scHPF/issues#fork-destination-box), make your
changes, and submit a pull request.

