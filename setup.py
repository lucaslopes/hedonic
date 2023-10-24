from setuptools import setup, find_packages

setup(
    name='hedonic',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'igraph',
        'numpy',
    ],
)