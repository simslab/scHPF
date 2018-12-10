#!/usr/bin/env python

from setuptools import find_packages, setup

__version__ = '0.2.1'

requires = ['scikit-learn',
            'numba',
            'scipy >= 1.1',
            'numpy',
            'pandas' # TODO get rid of this because only used in preprocessing
            ]

extras_require = {
        'loompy_preprocessing' : ['loompy']
        }

setup(
    name='scHPF',
    version=__version__,
    packages=find_packages(),
    scripts=['bin/scHPF'],
    python_requires='>=3.6',
    install_requires=requires,
    extras_require=extras_require,
    author = 'Hanna Mendes Levitin',
    author_email = 'hml2134@columbia.edu',
    description='Single-cell Hierarchical Poisson Factorization',
    license="BSD",
    url='https://www.github.com/simslab/scHPF',
)
