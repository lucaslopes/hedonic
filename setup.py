from setuptools import setup, find_packages

setup(
  name='hedonic',
  version='0.0.1',
  packages=find_packages(),
  install_requires=[
    'igraph',
    'ipython',
    'tqdm',
    'stopwatch.py',
    'pyarrow',
    'joblib',
    'numpy>=1.19.0',
    'pandas>=1.1.0',
    'networkx>=2.5',
    'cairocffi>=1.2.0',
    'scipy>=1.5.0',
    'matplotlib>=3.6.0',
    'plotly>=5.3.0',
  ],
)