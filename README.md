# Single-cell Hierarchical Poisson Factorization

## About
scHPF is a tool for _de novo_ discovery of both discrete and continuous expression patterns in single-cell RNA\-sequencing (scRNA-seq). We find that scHPF’s sparse low-dimensional representations, non-negativity, and explicit modeling of variable sparsity across genes and cells produce highly interpretable factors.

- [Documentation](https://schpf.readthedocs.io/en/latest/)
- [Paper at Molecular Systems Biology](http://msb.embopress.org/content/15/2/e8557.full.pdf)
- [Application to human tissue T cells across multiple donors and tissues](https://www.biorxiv.org/content/10.1101/555557v1) 

##  Installation
### Environment & Dependencies
scHPF requires Python >= 3.6 and the packages:
- numba ([version needed depends on Python version](https://schpf.readthedocs.io/en/latest/install.html#numba-compatibility), but should be safe with 0.45)
- scikit-learn
- pandas
- (optional) loompy

The easiest way to setup an environment for scHPF is with the Anaconda
Python distribution in [Miniconda](https://conda.io/miniconda.html) or
[anaconda](https://www.continuum.io/downloads):

```
conda create -n schpf_p37 python=3.7 scikit-learn numba=0.45 pandas

# for newer anaconda versions
conda activate schpf_p37
# XOR older anaconda verstions
source activate schpf_p37

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

## Quick Start: Command Line Interface

1. [Prepare your data](https://schpf.readthedocs.io/en/latest/prep-cli.html). 

2. [Train a model](https://schpf.readthedocs.io/en/latest/train-cli.html).

3. [Get gene and cell scores](https://schpf.readthedocs.io/en/latest/score-cli.html)


## API
scHPF has a scikit-learn like API. Trained models are stored in a serialized
joblib format.


## Help and support
If you have any questions/errors/issues, please [open an issue](https://github.com/simslab/scHPF/issues/new) 
and I be happy to to provide whatever help and guidance I can.


## Contributing
Contributions to scHPF are welcome. Please get in touch if you would like to
discuss/check it's something I've already done but haven't pushed to master yet.
To contribute, please [fork
scHPF](https://github.com/simslab/scHPF/issues#fork-destination-box), make your
changes, and submit a pull request.

##  References
Hanna Mendes Levitin, Jinzhou Yuan, Yim Ling Cheng, Francisco JR Ruiz, Erin C Bush, 
Jeffrey N Bruce, Peter Canoll, Antonio Iavarone, Anna Lasorella, David M Blei, Peter A Sims.
__"*De novo* gene signature identification from single‐cell RNA‐seq with hierarchical Poisson 
factorization."__ Molecular Systems Biology, 2019. [[Open access article]](http://msb.embopress.org/content/15/2/e8557.full.pdf)

Peter A. Szabo\*, Hanna Mendes Levitin\*, Michelle Miron, Mark E. Snyder, Takashi Senda, 
Jinzhou Yuan, Yim Ling Cheng, Erin C. Bush, Pranay Dogra, Puspa Thapa, Donna L. Farber, 
Peter A. Sims. __"A single-cell reference map for human blood and tissue T cell 
activation reveals functional states in health and disease."__ In press, 2019. 
[[preprint]](https://www.biorxiv.org/content/10.1101/555557v1)
\* Co-first authors

