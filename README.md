# Single-cell Hierarchical Poisson Factorization

Pre-release of [Single-cell Hierarchical Poisson Factorization (scHPF)](https://www.biorxiv.org/content/early/2018/07/11/367003), as described in the forthcoming manuscript: <br/> *De novo* Gene Signature Identification from Single-Cell RNA-Seq with Hierarchical Poisson Factorization.

scHPF is a tool for _de novo_ discovery of both discrete and continuous expression patterns in single-cell RNA\-sequencing (scRNA-seq) data. It adapts [Hierarchical Poisson Factorization](http://www.cs.columbia.edu/~blei/papers/GopalanHofmanBlei2015.pdf) to avoid prior normalization and model variable sparsity across genes and cells. Algorithmic details, benchmarking against alternative methods, and scHPF's application to a spatially sampled high-grade glioma can be found in our [paper on biorXiv](https://www.biorxiv.org/content/early/2018/07/11/367003).

# Documentation
Info for numba version forthcoming


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
