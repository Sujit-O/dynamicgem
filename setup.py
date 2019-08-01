#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools
import os
# get __version__ from _version.py
ver_file = os.path.join('dynamicgem', '_version.py')

with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'dynamicgem'

INSTALL_REQUIRES = [i.strip() for i in open("requirements.txt").readlines()]

with open("README.md", "r") as fh:
    long_description = fh.read()

VERSION = __version__

setuptools.setup(
    name='dynamicgem',
    version=VERSION,
    author="Sujit Rokka Chhetri, Palash Goyal, Martinez Canedo, Arquimedes",
    author_email="sujitchhetri@gmail.com",
    description="A Python library for Dynamic Graph Embedding Methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sujit-O/dynamicgem.git",
    packages=setuptools.find_packages(exclude=['dataset', 'py-env', 'build', 'dist', 'venv','intermediate','output','dynamicgem.egg-info']),
    package_dir={DISTNAME: 'dynamicgem'},
    setup_requires=['sphinx>=2.1.2'],
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)