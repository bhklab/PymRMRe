import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pymrmre-bhaibeka",
    version="0.0.1",
    author='Bo Li, Benjamin Haibe-Kains',
    author_email="benjamin.haibe.kains@utoronto.ca",
    description="A Python package for Parallelized Minimum Redundancy, Maximum Relevance (mRMR) Ensemble Feature selections.",
    long_description=long_description,
    long_description_contenct_type="text/markdown",
    url="https://github.com/bhklab/PymRMRe",
    ext_modules=[Extension("cymrmre",
                [".c", ".cpp",
                include_dirs=["cymrmre/include"]]
                )]
    packages=setuptools.find_packages(),
    keywords='pharmacogenomics mrmr minimumredundacymaximumrelevance',
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License:: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'cython'
    ],
    python_requires='>=3.6'
)