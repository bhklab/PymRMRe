from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np
import glob as gg
import os
import sys

with open('README.md') as fh:
      readme = fh.read()

expt = ["expt.pyx", "Matrix.cpp", "Data.cpp", "MutualInformationMatrix.cpp", "Filter.cpp", "MathExtensions.cpp"]

requirements = [
    'numpy',
    'pandas',
    'scipy'
]

if sys.platform == 'darwin':
    extensions = Extension(
    "expt", 
    sources = ["pymrmre/expt/" + str for str in expt], 
    language = "c++",
    include_dirs=[np.get_include()],
    extra_compile_args = ["-stdlib=libc++"],
    extra_link_args = ["-stdlib=libc++"],
    ) 
else:
    extensions = Extension(
    "expt", 
    sources = ["pymrmre/expt/" + str for str in expt], 
    language = "c++",
    #include_dirs=[np.get_include()],
    )

setup(
    name = "pymrmre",
    version="1.0.5",
    description="A Python package for Parallelized Minimum Redundancy, Maximum Relevance (mRMR) Ensemble Feature selections.",
    long_description=readme,
    long_description_content_type='text/markdown',
    url="https://github.com/bhklab/PymRMRe",
    author="Bo Li, Benjamin Haibe-Kains",
    author_email="benjamin.haibe.kains@utoronto.ca",
    test_suite = 'tests',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    setup_requires=[
        'cython>=0.25'
    ],
    license='MIT license',
    zip_safe=False,
    keywords='pymrmre featureselection genomics computationalbiology',
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()]
)
