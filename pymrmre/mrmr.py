import numpy as np
import pandas as pd
import os

def mrmr_selection(features : pd.DataFrame,
                   target_features : list,
                   feature_types : list,
                   solution_feature_count : int,
                   solution_count : int = 1,
                   fixed_feature_count : int = 0,
                   method : str = 'exhaustive',
                   estimator : str = 'pearson'):

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
    if fixed_feature_count > solution_feature_count:
        raise Exception("The count of fixed-selected features should be less than the expected count of one solution")

    
    # No need for data pre-processing, since the data frame input is already the combined version
    ## No need to specify the survival option, since no matter what the target indices have to 
    ## be the input
    '''
    if not survival:
        features['time'] = features['target']
        features['event'] = 1
    '''
    
    #features.set_index('id', replace = True)
    features = features.infer_objects()

    # Build the mRMR data
    mrmr_data = MrmreData(data = features, 
                          feature_types = feature_types)

    # Find the target indices and fixed features selected
    '''
    target_indices = []
    for tf in target_features: 
        target_indices.append(features.columns.get_loc(tf))
    '''
    target_indices = target_features
    #fixed_selected_indices = list(range(fixed_selected_count))
    
    # Build the mRMR Filter
    levels = [solution_count] + [1] * (solution_feature_count - fixed_feature_count - 1)
    mrmr_filter = MrmreFilter(data = mrmr_data, 
                              target_indices = target_indices, 
                              fixed_feature_count = fixed_feature_count,
                              levels = levels)

    solutions, indices = [], []
    mrmr_solutions = mrmr_filter.solutions()
    for key, value in mrmr_solutions.items():
        result = []
        indices.append(key)
        for col in range(value.shape[1]):
            result.append(list(value[:, col]))
            if fixed_feature_count > 0:
                result[-1] = list(range(fixed_feature_count)) + result[-1]
        solutions.append(result)
    
    solutions = pd.Series(solutions)
    solutions.index = indices

    return solutions

def mrmr_selection_with_clinical(target_df : pd.DataFrame,
                   features_df : pd.DataFrame,
                   target_features : list,
                   feature_count : int,
                   solution_count : int = 1,
                   method : str = 'exhaustive',
                   estimator : str = 'pearson',
                   survival : bool = False):
    # Data Pre-processing
    target = target_df.copy()
    if not survival:
        target['time'] = target['target']
        target['event'] = 1
    
    target.set_index('id', inplace = True)
    features = features_df.merge(target[['time', 'event']], how = 'inner', left_index = True, right_index = True)
    features = features.infer_objects()

    # Build the mRMR Data
    mrmr_data = MrmreData(data = features)
    
    # Find the target_indices
    target_indices = []
    for feature in target_features:
        target_indices.append(features.columns.get_loc(feature))

    # Build the mRMR Filter
    levels = [solution_count] + [1] * features_count
    mrmr_filter = MrmreFilter(data = mrmr_data, target_indices = target_indices, levels = levels)
    
    return mrmr_filter._filter
    

'''
def run_mrmr_selections(features : pd.DataFrame,
                        solution_count : int = 1,
                        features_count : int,
                        target_indices : np.array):

    mrmr_data = MrmreData(data = features)
    levels = [solution_count] + [1] * features_count
    mrmr_filter = MrmreFilter(data = mrmr_data, target_indices = target_indices, levels = np.array(levels))
    selected_features = mrmr_filter._filter
    return selected_features.iloc[:, 0]
    

def mrmr_selection(features : pd.DataFrame,
                   target_df : pd.DataFrame,
                   solution_count : int,
                   feature_count : int,
                   survival : bool = false):
    
    target = target_df.copy()
    # Merge the features and labels
    if not survival:
        target['time'] = target['target']
        target['event'] = 1

    features = features.merge(target[['time', 'event']], how = 'inner', left_index = True, right_index = True)
    mrmr_list = run_mrmr_selections(features, solution_count, features_count, 'time')
    return mrmr_list
'''

    

