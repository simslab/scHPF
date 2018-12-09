import setuptools

setuptools.setup(
  name='scHPF',
  description='Single-cell Hierarchical Poisson Factorization',
  version='0.1',
  url='https://www.github.com/simslab/scHPF',
  license='GPLv3',
  install_requires=[
    'numpy',
    'pandas',
    'pyyaml',
    'tensorflow',
  ],
  extras_require={
    'plotting': ['seaborn']
  },
  packages=setuptools.find_packages(),
)
