#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
from .MrmreData import *
from .MrmreFilter import *


def mrmr_ensemble(features : pd.DataFrame,
                  targets : pd.DataFrame,
                  solution_length : int,
                  solution_count : int = 1,
                  category_features : list = [],
                  fixed_features : list = [],
                  method : str = 'exhaustive',
                  estimator : str = 'pearson',
                  return_index : bool = False,
                  return_with_fixed : bool = True):

    '''
    :param features: Pandas dataframe, the input dataset
    :param targets: Pandas dataframe, the target features
    :param fixed_features: List, the list of fixed features (column names)
    :param category_features: List, the list of features of categorical type (column names)
    :param solution_length: Integer, the number of features contained in one solution
    :param solution_count: Integer, the number of solutions to be returned
    :param method: String, the different ways to run the algorithm, exhaustive or bootstrap
    :param estimator: String, the way of computing continuous estimators
    :param return_index: Boolean, to determine whether the solution contains the indices or column names of selected features
    :param return_with_fixed: Boolean, to determine whether the solution contains the fixed selected features
    :return: The pandas series, the solutions of selected features
    '''
    
    ## Handle some corner cases
    # The input features type
    
    # The method and estimator do not match
    if method not in ["exhaustive", "bootstrap"]:
        raise Exception('The method must be exhaustive or bootstrap')
    
    if estimator not in ["pearson", "spearman", "kendall", "frequency"]:
        raise Exception("The continuous estimator must be chosen from pearson, spearman, kendall and frequency")

    # The length of one features larger than fixed one
    if len(fixed_features) > solution_length:
        raise Exception("The count of fixed-selected features should be less than the expected count of one solution")

    fixed_feature_count = len(fixed_features)

    if fixed_feature_count == 0:
        return_with_fixed = False

    ## Combine the features dataframe with targets
    if features.shape[0] != targets.shape[0]:
        raise Exception("The dimension of targets do not match with features dataframe")
    
    features = features.join(targets)
    features = features.infer_objects()
    target_indices = []

    for x in targets.columns:
        target_indices.append(features.columns.get_loc(x))
    
    ## Reorder the input data
    for x in fixed_features:
        if x not in features.columns:
            raise Exception('Some of fixed features are not in input data')
    
    non_fixed_features = [x for x in features.columns if x not in fixed_features]
    features = features.reindex(columns=fixed_features + non_fixed_features)

    ## Build the feature types
    feature_types = [0] * features.shape[1]

    for x in category_features:
        feature_types[features.columns.get_loc(x)] = 1
        print(feature_types)

    ## Build the mRMR data
    mrmr_data = MrmreData(data = features, 
                          feature_types = feature_types)

    # Error for a case known to break c_estimate_filters in the C++ code
    if (len(fixed_features) + len(category_features)) > features.shape[1] - 1:
        raise Exception('This function does not work when there are only fixed'
                        'and categorical features')

    # Find the target indices and fixed features selected

    ## Build the mRMR Filter
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
                 fixed_features : list = [],
                 method : str = 'exhaustive',
                 estimator : str = 'pearson',
                 return_index : bool = False):
    
    return mrmr_ensemble(features=features, target_features=target_features, fixed_features= fixed_features,
                         feature_types = feature_types, solution_length = solution_length, solution_count = 1, 
                         method = method, estimator = estimator, return_index = return_index)

def mrmr_ensemble_survival(features : pd.DataFrame,
                           targets : pd.DataFrame,
                           solution_length : int,
                           solution_count : int = 1,
                           category_features : list = [],
                           fixed_features : list = [],
                           method : str = 'exhaustive',
                           estimator : str = 'pearson',
                           return_index : bool = False,
                           return_with_fixed : bool = True):
        
    # The method and estimator do not match
    if method not in ["exhaustive", "bootstrap"]:
        raise Exception('The method must be exhaustive or bootstrap')
    
    if estimator not in ["pearson", "spearman", "kendall", "frequency"]:
        raise Exception("The continuous estimator must be chosen from pearson, spearman, kendall and frequency")

    # The length of one features larger than fixed one
    if len(fixed_features) > solution_length:
        raise Exception("The count of fixed-selected features should be less than the expected count of one solution")

    # The survival data should have two columns 
    # If the target is survival data, the target should be have two columns (event and time)
    if targets.shape[1] != 2:
        raise Exception("The survial data should be two columns (event and time)")

    # The event and time
    event, time = targets.columns[0], targets.columns[1]



    fixed_feature_count = len(fixed_features)

    ## Combine the features dataframe with survival data
    if features.shape[0] != targets.shape[0]:
        raise Exception("The dimension of targets do not match with features dataframe")

    features = features.join(targets)
    features = features.infer_objects()

    ## Reorder the input data (after merge)
    
    for x in fixed_features:
        if x not in features.columns:
            raise Exception('Some of fixed features are not in input data')

    if fixed_feature_count != 0:
        ## Some problems may exist here
        #fixed_features.append(fixed_features.pop(fixed_features.index(event)))
        non_fixed_features = [x for x in features.columns if x not in fixed_features and x != time and x != event]
        features = features.reindex(columns = fixed_features + [event, time] + non_fixed_features)
    else:
        ## Some problems may exist here
        non_surv_features = [x for x in features.columns if x != time and x != event]
        mid = len(non_surv_features) // 2
        features = features.reindex(columns = non_surv_features[0:mid] + [event,time] + non_surv_features[mid:])
        #features = features.reindex(columns = non_surv_features[0:8] + [event,time] + non_surv_features[8:])

    target_indices = [features.columns.get_loc(time)] 

    ## Build the feature types
    feature_types = [0] * features.shape[1]

    feature_types[features.columns.get_loc(event)] = 2
    feature_types[features.columns.get_loc(time)] = 3
    
    for x in category_features:
        feature_types[features.columns.get_loc(x)] = 1
    

    ## Build the mRMR data
    mrmr_data = MrmreData(data = features, 
                          feature_types = feature_types)
    
    ## Build the mRMR Filter
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

    





    

    