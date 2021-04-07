# PymRMRe

## Description
Feature selection is one of the main challenges in analyzing high-throughput genomic data. Minimum redundancy maximum relevance (mRMR) is a particularly fast feature selection method for finding a set of both relevant and complementary features. The Pymrmre package, extend the mRMR technique by using an ensemble approach to better explore the feature space and build more robust predictors. To deal with the computational complexity of the ensemble approach, the main functions of the package are implemented and parallelized in C++ using openMP Application Programming Interface. The package also supports making best selections with some fixed-selected features.

## Prerequisite
`
Python(>=3.6.0)
`
<br>
`
Cython(>=0.29.12)
`
<br>
`
numpy(>=1.16.4)
`
<br>
`
pandas(>=0.25.0)
`


## Installation
`
pip install Pymrmre
`

## Insturctions

Two primary functions are provided in this package currently:

* mrmr_ensemble: It provides the ensemble (multiple) solutions of feature selection given the input of feature dataset and target column, it supports the feature selection with preselection as well. 
  *  :param *features*: Pandas dataframe, the input dataset
  *  :param *targets*: Pandas dataframe, the target features
  *  :param *fixed_features*: List, the list of fixed features (column names), the default is empty list
  *  :param *category_features*: List, the list of features whose types are categorical (column names), the default is empty list
  *  :param *solution_length*: Integer, the number of features contained in one solution
  *  :param *solution_count*: Integer, the number of solutions to be returned, the default is 1
  *  :param *estimator*: String, the way of computing continuous estimators, the default is Pearson
  *  :param *return_index*: Boolean, to determine whether the solution contains the indices or column names of selected features, the default is False
  *  :param *return_with_fixed*: Boolean, to determine whether the solution contains the fixed selected features, the default is True
  *  :return: Pandas series, the solutions of selected features

Example code:

`
import pandas as pd
`
<br>
`
from pymrmre import mrmr
`

Load the input data and target variable, suppose for input X we have ten features (*f1*, *f2*, ..., *f10*):

`
X = pd.read_csv('train_x.csv')
`
<br>
`
Y = pd.read_csv('train_y.csv')
`
<br>

Suppose we want to generate 3 solutions, where each solution has 5 features. We want to see *f1* exists in all solutions (preselection), and we know that *f4* and *f5* are categorical variables as well, the code should be like this:

`
solutions = mrmr.mrmr_ensemble(features=X,targets=Y,fixed_features=['f1'],category_features=['f4','f5'],solution_length=5,solution_count=3)
`
<br>

Because the solution we generated is of the type Pandas series, which has the target variable name as column header. To access the contents of all three solutions, the code is like this:

`
solutions.iloc[0]
`
<br>

To access one of the solutions, the code is like this (i is 0 - 2 here since we generate 3 solutions here):

`
solutions.iloc[0][i]
`
<br>


