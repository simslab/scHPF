#!/usr/bin/env python

from setuptools import find_packages, setup

# get version from file
__version__ = '0.0.0'
exec(open('schpf/_version.py').read())

requires = ['scikit-learn',
            'numba >=0.35, <=0.40',
            'scipy >= 1.1',
            'numpy',
            'pandas'
            ]

tests_require = ['pytest']
extras_require = {
        'loompy' : ['loompy'],
        'docs' : ['sphinx-argparse']
        }

setup(
    name='scHPF',
    version=__version__,
    packages=find_packages(),
    scripts=['bin/scHPF'],
    python_requires='>=3.6',
    install_requires=requires,
    tests_require=tests_require,
    extras_require=extras_require,
    author = 'Hanna Mendes Levitin',
    author_email = 'hml2134@columbia.edu',
    description='Single-cell Hierarchical Poisson Factorization',
    license="BSD",
    url='https://www.github.com/simslab/scHPF',
)
