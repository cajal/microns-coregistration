#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='microns-nda',
    version='0.0.1',
    description='EM/ Functional coregistration for MICrONS',
    author='Stelios Papadopoulos',
    author_email='spapadop@bcm.edu',
    packages=find_packages(exclude=[]),
    install_requires=['numpy', 'scipy', 'tqdm', 'pandas', 'seaborn', 'matplotlib', 'torch']
)