import numpy as np 
import pandas as pd 
from scipy.special import comb

class MrmreFilter:

    def __init__(self,
                 data : MrmreData,
                 prior_weight : np.array,
                 target_indices : np.array,
                 levels : np.array,
                 method : str,
                 continous_estimator : str,
                 outX : bool,
                 bootstrap_count : int):
        self._method = method 
        self._continous_estimator = continuous_estimator

        if type(data) != 'mRMRe_data':
            raise Exception('data must be of type mRMRe_data')
        
        if len(data._prior) != 0:
            if not prior_weight:
                raise Exception('prior weight must be provided if there are priors')
            elif prior_weight < 0 or prior_weight > 1:
                raise Exception('prior weight must be a value ranging from 0 to 1')
        else:
            prior_weight = 0

        ## Target processing

        if any(x < 1 for x in target_indices) or any(x > data.featureCount for x in target_indices):
            raise Exception('target indices must only contain values ranging from 1 to the number of features in data')
        
        ## Level processing

        if not levels:
            raise Exception('levels must be provided')

        self._target_indices = target_indices.astype(int)
        self._levels = levels.astype(int)

        target_indices = data.expandFeatureIndices(target_indices).astype(int) - 1

        ## Filter; Mutual Information and Causality Matrix

        mi_matrix = np.zeros((data._nrow, data_ncol))

        if method == 'exhaustive':

            ## Level processing
            if np.prod(levels) > comb(data._featureCount - 1, len(levels)):
                raise Exception('user cannot request for more solutions than is possible given the data set')

            '''
            Call the cpp functions 
            '''
            result 
        elif method == 'bootstrap':
            '''
            Call the cpp functions
            '''
            result 
        else:
            raise Exception('Unrecognized method: use exhaustive or bootstrap')


        ## 
        self._filters = 

    

    def sampleCount(self):

        return len(self._sample_names)

    def sampleNames(self, data):

        return data._sample_names

    def featureCount(self):

        return len(self._feature_names)

    def featureNames(self, data):

        return data._feature_names

    def solutions(self, mi_threshold = -float('inf'), causality_threshold = float('inf')):
        # filters[target][solution, ] is a vector of selected features
        # in a solution for a target; missing values denote removed features

        