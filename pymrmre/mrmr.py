#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
from MrmreData import *
from MrmreFilter import *

def mrmr_ensemble(features : pd.DataFrame,
                  target_features : list,
                  feature_types : list,
                  solution_length : int,
                  solution_count : int = 1,
                  fixed_feature_count : int = 0,
                  method : str = 'exhaustive',
                  estimator : str = 'pearson',
                  return_index : bool = False,
                  return_with_fixed : bool = True):

    '''
    :param features: Pandas dataframe, the input dataset
    :param target_features: List, the list of target features, could be column names of indices
    :param feature_types: List, the feature types (continuous, discrete, survival) of all columns in features
    :param solution_length: Integer, the number of features contained in one solution
    :param solution_count: Integer, the number of solutions to be returned
    :param fixed_feature_count: Integer, the number of features to be fixed in solutions
    :param method: String, the different ways to run the algorithm, exhaustive or bootstrap
    :param estimator: String, the way of computing continuous estimators
    :param return_index: Boolean, to determine whether the solution contains the indices or column names of selected features
    :param return_with_fixed: Boolean, to determine whether the solution contains the fixed selected features
    :return: The pandas series, the solutions of selected features
    '''
    
    ## Handle some corner cases
    # The input features type
    if len(feature_types) != features.shape[1]:
        raise Exception('The count of feature types does not match with the size of dataset')
    
    if any((x < 0 or x > 3) for x in feature_types):
        raise Exception('The feature type should within the range 0, 1, 2, 3')
    
    # The method and estimator do not match
    if method not in ["exhaustive", "bootstrap"]:
        raise Exception('The method must be exhaustive or bootstrap')
    
    if estimator not in ["pearson", "spearman", "kendall", "frequency"]:
        raise Exception("The continuous estimator must be chosen from pearson, spearman, kendall and frequency")

    # The length of one features larger than fixed one
    if fixed_feature_count > solution_length:
        raise Exception("The count of fixed-selected features should be less than the expected count of one solution")

    if fixed_feature_count <= 0:
        fixed_feature_count = 0
        return_with_fixed = False

    
    #features.set_index('id', replace = True)
    features = features.infer_objects()

    # Build the mRMR data
    mrmr_data = MrmreData(data = features, 
                          feature_types = feature_types)

    # Find the target indices and fixed features selected
    target_indices = []
    if type(target_features[0]) == str:
        for tf in target_features: 
            target_indices.append(features.columns.get_loc(tf))
    elif type(target_features[0]) == int:
        target_indices = target_features
    else:
        raise Exception("The target feature input should be string (feature name) or integer (indices)")

    # Build the mRMR Filter
    levels = [solution_count] + [1] * (solution_length - fixed_feature_count - 1)
    mrmr_filter = MrmreFilter(data = mrmr_data, 
                              method = method,
                              target_indices = target_indices, 
                              fixed_feature_count = fixed_feature_count,
                              levels = levels)
    

    feature_names = list(features.columns.values)
    
    solutions, indices = [], []
    mrmr_solutions = mrmr_filter.solutions()


    if return_index:
        for key, value in mrmr_solutions.items():
            result = []
            indices.append(key)
            for col in range(value.shape[1]):
                result.append(list(value[:, col]))
                if fixed_feature_count > 0 and return_with_fixed:
                    result[-1] = list(range(fixed_feature_count)) + result[-1]
            
            solutions.append(result)
    
    else:
        def find_feature_names(list_features : list):
            result = []
            for f in list_features:
                result.append(feature_names[f])
            return result

        for key, value in mrmr_solutions.items():
            result = []
            indices.append(feature_names[key])
            for col in range(value.shape[1]):
                result.append(list(value[:, col]))
                if fixed_feature_count > 0 and return_with_fixed:
                    result[-1] = list(range(fixed_feature_count)) + result[-1]
                result[-1] = find_feature_names(result[-1])
            
            solutions.append(result)
    
    solutions = pd.Series(solutions)
    solutions.index = indices
    
    return solutions


def mrmr_classic(features: pd.DataFrame,
                 target_features : list,
                 feature_types : list,
                 solution_length : int,
                 fixed_feature_count : int = 0,
                 method : str = 'exhaustive',
                 estimator : str = 'pearson',
                 return_index : bool = False):
    
    return mrmr_ensemble(features = features, target_features = target_features, feature_types = feature_types,
                        solution_length = solution_length, fixed_feature_count = fixed_feature_count, 
                        solution_count = 1, method = method, estimator = estimator, return_index = return_index)

