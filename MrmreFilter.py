import numpy as np 
import pandas as pd 
import constants
#import expt
from src.expt import export_filters
from src.expt import export_mim
from MrmreData import *
from constants import *
from scipy.special import comb

class MrmreFilter:

    def __init__(self,
                 data : MrmreData = None,
                 prior_weight : float = None,
                 target_indices : list = [0],
                 levels : list = [0],
                 method : str = 'exhaustive',
                 continuous_estimator : str = 'pearson',
                 outX : bool = True,
                 bootstrap_count : int = 0):

        ## Declare the private or protected variables here
        self._method = method
        self._continuous_estimator = MAP.estimator_map[continuous_estimator]
        self._filter = pd.Series()
        self._scores = pd.Series()
        self._causality_list = pd.Series()
        self._feature_names = data.featureNames()
        self._sample_names = data.sampleNames()

        '''
        if type(data) != 'MrmreData':
            raise Exception('data must be of type MrmreData')
        '''
        
        if data._priors and len(data._priors) != 0:
            if not prior_weight:
                raise Exception('prior weight must be provided if there are priors')
            elif prior_weight < 0 or prior_weight > 1:
                raise Exception('prior weight must be a value ranging from 0 to 1')
        else:
            prior_weight = 0.0

        ## Target processing

        if any(x < 0 for x in target_indices) or any(x > data.featureCount() - 1 for x in target_indices):
            raise Exception('target indices must only contain values ranging from 0 to the number of features minus one in data')
        
        # This is because sometimes we accept the column names as inputs (the feature names)
        # But I will process this before
        self._target_indices = np.array(target_indices).astype(int)
        #self._target_indices = target_indices.astype(int)
        ## Level processing

        if len(levels) == 0:
            raise Exception('levels must be provided')
        
        #self._levels = levels.astype(int)
        self._levels = np.array(levels).astype(int)

        # The index 0/1 problems?
        target_indices = data._expandFeatureIndices(self._target_indices + 1).astype(int)

        ## Filter; Mutual Information and Causality Matrix
        # Mutual Information matrix
        mi_matrix = np.empty((data.sampleCount(), data.featureCount())).astype(np.double)
        mi_matrix[:] = np.nan


        ## Check the input data for debugging
       

        if method == 'exhaustive':

            ## Level processing
            if np.prod(levels) - 1 > comb(data.featureCount() - 1, len(levels)):
                raise Exception('user cannot request for more solutions than is possible given the data set')

            res = export_filters(self._levels.astype(np.int32),
                                 data._data.values.flatten('F'),
                                 np.array([]),
                                 prior_weight,
                                 data._strata.values.astype(np.int32),
                                 data._weights.values,
                                 data._feature_types.values.astype(np.int32),
                                 data._data.shape[0],
                                 data._data.shape[1],
                                 len(data._strata.unique()),
                                 self._target_indices.astype(np.uint32),
                                 self._continuous_estimator,
                                 int(outX == True),
                                 bootstrap_count,
                                 mi_matrix.flatten())
        else:
            raise Exception('Unrecognized method: use exhaustive or bootstrap')


        ## After building the result, result is the data structure of Result defind in exports.h
        ## The returned filter is object of list
        ## The returned filter need to use target lists as name
        filters = res[0]              # List<List<int>>
        causality_list = res[1][0]      # List<List<float>>
        scores = res[1][1]              # List<List<float>>

        print(filters)
        print(causality_list)
        print(scores)
        
        # Build the filter based on solutions
        _filters = []
        for sol in filters:
            _sol = data._compressFeatureIndices(sol + 1).reshape(len(self._levels), np.prod(self._levels))
            _filters.append(_sol)
        
        self._filter = pd.Series(_filters)
        self._filter.index = target_indices
        
        # Build the causality list
        _causality_list = []
        _, unique_indices = np.unique(data._compressFeatureIndices(list(range(data.sampleCount())), return_index = True))
        for cas in causality_list:
            _cas = cas[unique_indices]
            _causality_list.append(_cas)
        
        self._causality_list = pd.Series(_causality_list)
        self._causality_list.index = target_indices

        # Build the scores matrix
        _scores = []
        for sc in scores:
            _sc = sc.reshape(len(self._levels), np.prod(self._levels))
            _scores.append(_sc)
        
        self._scores = pd.Series(_scores)
        self._scores.index = target_indices
        
        # Build the mutual information matrix
        self._mi_matrix = data._compressFeatureMatrix(mi_matrix.reshape(data.sampleCount(), data.featureCount()))
        print(self._mi_matrix)
        

    def sampleCount(self):

        return len(self._sample_names)

    def sampleNames(self, data):

        return data._sample_names

    def featureCount(self):

        return len(self._feature_names)

    def featureNames(self, data):

        return data._feature_names

    def solutions(self, mi_threshold = -float('inf'), causality_threshold = float('inf')):
        ## filters[target][solution, ] is a vector of selected features
        ## in a solution for a target; missing values denote removed features
        _filters = []
        for target_index in self._target_indices:
            result_matrix = self._filters.loc[target_index]
            causality_dropped, _ = np.where(np.array(self._causality_list.loc[target_index]) > causality_threshold)
            mi_dropped, _ = np.where(-.5 * np.log(1 - np.square(self._mi_matrix[:, target_index])) < mi_threshold)
            # Nan operation
            dropped = set(list(causality_dropped) + list(mi_dropped))
            for i, result in enumerate(result_matrix):
                if result in dropped:
                    result_matrix[i] = np.nan
            
            pre_return_matrix = np.flip(result_matrix, axis = 0)
            _filters.append(pre_return_matrix)
        
        _filters = pd.Series(_filters)
        _filters.index = self._target_indices

        return _filters

    def scores(self):
        mi_matrix = self.mim()
        target_indices = self._target_indices
        scores = pd.Series()
        solutions = self._solutions()
        for i, target in enumerate(target_indices):
            # 
            sub_solution = solutions.loc(target)   # The sub_solution is matrix(np.array)
            sub_score_target = np.array()
            for col in range(sub_solution.shape[1]):
                sub_score = list()
                previous_features_mean = list()
                for j in range(sub_solution[:, col].shape[0]):
                    feature_j = sub_solution[:, col][j]
                    if j == 0:
                        sub_score.append(mi_matrix[target, feature_j])
                        continue
                    previous_features_mean.append(mi_matrix[target, sub_solution[:, col][j - 1]])
                    ancestry_score = sum(previous_features_mean) / len(previous_features_mean)
                    sub_score.append(mi_matrix[target, feature_j] - ancestry_score)
                sub_score_target = np.hstack(sub_score_target, np.array(sub_score))
            
            scores = scores.append(pd.Series([sub_score_target]))
        
        scores.index = target_indices
        
        return scores


    def mim(self, method):
        ## The method should be within mi or cor
        return self._mi_matrix

    def target(self):
        return self._target_indices

        