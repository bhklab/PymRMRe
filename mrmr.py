#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
from MrmreData import *
from MrmreFilter import *

def mrmr_selection(features : pd.DataFrame,
                   target_features : list,
                   target_count : int,
                   solution_count : int = 1,
                   fixed_selected_count : int = 0,
                   method : str = 'exhaustive',
                   estimator : str = 'pearson',
                   survival : bool = False):
    # No need for data pre-processing, since the data frame input is already the combined version
    if not survival:
        features['time'] = features['target']
        features['event'] = 1
    
    features.set_index('id', replace = True)
    features = features.infer_objects()

    # Build the mRMR data
    mrmr_data = MrmreData(data = features)

    # Find the target indices and fixed features selected
    target_indices = []
    for tf in target_features:
        target_indices.append(features.columns.get_loc(tf))

    fixed_selected_indices = list(range(fixed_selected_count))
    
    # Build the mRMR Filter
    levels = [solution_count] + [1] * target_count
    mrmr_filter = MrmreFilter(data = mrmr_data, target_indices = target_indices, fixed_selected_count = fixed_selected_count,
                              levels = levels)
    
    return mrmr_filter._filter



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

    

