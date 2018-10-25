import setuptools

setuptools.setup(
    name='scHPF',
    description='Single-cell Hierarchical Poisson Factorization',
    version='0.2',
    url='https://www.github.com/simslab/scHPF',
    license='GPLv3',
    install_requires=[
        'numpy',
        'scikit-learn',
        'numba'
    ],
    extras_require={
    },
    packages=setuptools.find_packages(),
)
