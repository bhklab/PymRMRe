import numpy as np 
import pandas as pd 
import constants
import expt
from scipy.special import comb

class MrmreFilter:

    def __init__(self,
                 data : MrmreData = None,
                 prior_weight : float = None,
                 target_indices : np.array = None,
                 levels : np.array = None,
                 method : str = 'bootstrap',
                 continous_estimator : str = 'pearson',
                 outX : bool = True,
                 bootstrap_count : int = 0):

        ## Declare the private or protected variables here
        self._estimator_map = {'pearson'  : constants.ESTIMATOR.PEARSON, 
                               'spearman' : constants.ESTIMATOR.SPEARMAN, 
                               'kendall'  : constants.ESTIMATOR.KENDALL,
                               'frequency': constants.ESTIMATOR.FREQUENCY}
        self._method = method
        self._continous_estimator = self._estimator_map[continous_estimator]
        self._filter = pd.Series()
        self._scores = pd.Series()
        self._causality_list = pd.Series()
        self._feature_names = data._featureNames()
        self._sample_names = data._sampleNames()

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

        if any(x < 1 for x in target_indices) or any(x > data.featureCount() for x in target_indices):
            raise Exception('target indices must only contain values ranging from 1 to the number of features in data')
        
        self._target_indices = target_indices.astype(int)
        ## Level processing

        if not levels:
            raise Exception('levels must be provided')
        
        self._levels = levels.astype(int)

        # The index 0/1 problems?
        target_indices = data._expandFeatureIndices(target_indices).astype(int) - 1

        ## Filter; Mutual Information and Causality Matrix
        # Nan operation here ?
        #mi_matrix = np.empty([data._nrow, data._ncol])
        mi_matrix = np.zeros(data._nrow, data._ncol)
        mi_matrix.fill(np.nan)

        if method == 'exhaustive':

            ## Level processing
            if np.prod(levels) - 1 > comb(data._featureCount - 1, len(levels)):
                raise Exception('user cannot request for more solutions than is possible given the data set')

            res = expt.export_filters(self._levels，
                                      data._data.values.flatten(),
                                      data._priors,
                                      prior_weight,
                                      data._strata,
                                      data._weights,
                                      data._feature_types,
                                      data._data.shape[0],
                                      data._data.shape[1],
                                      len(data._strata.unique()),
                                      target_indices,
                                      self._estimator_map[continuous_estimator],
                                      int(outX == true),
                                      bootstrap_count,
                                      mi_matrix)
            '''
            Call the cpp functions 
            '''

            #result 
        '''
        elif method == 'bootstrap':
            
            #Call the cpp functions
            
            result 
        '''
        else:
            raise Exception('Unrecognized method: use exhaustive or bootstrap')


        ## After building the result, result is the data structure of Result defind in exports.h
        ## The returned filter is object of list
        ## The returned filter need to use target lists as name
        solutions = res[0]              # List<List<int>>
        causality_list = res[1][0]      # List<List<float>>
        scores = res[1][1]              # List<List<float>>
        
        # Build the filter based on solutions
        for i, sol in enumerate(solutions):
            sol_matrix = data._compressFeatureIndices(sol + 1).reshape(len(self._levels), np.prod(self._levels))
            #self._filter.set(target_indices[i], [sol_matrix]) # Also set the index 
            self._filter = self._filter.append(pd.Series([sol_matrix]))
        
        self._filter.index = target_indices
        
        # Build the causality list
        _, cols_unique = np.unique(data._compressFeatureIndices(list(range(data._ncol))), return_index=true)
        for i, causality_array in enumerate(causality_list):
            causality_array = causality_array[cols_unique]
            #self._causality_list.set(target_indices[i], [causality_array])
            self._causality_list = self._causality_list.append(pd.Series([causality_array]))
        
        self._causality_list.index = target_indices

        # Build the scores matrix
        for i, score in enumerate(scores):
            sc_matrix = score.reshape(len(self._levels), np.prod(self._levels))
            #self._scores.set(target_indices[i], [sc_matrix])
            self._scores = self._scores.append(pd.Series([sc_matrix]))

        self._scores.index = target_indices
        
        # Build the mutual information matrix
        self._mi_matrix = data._compressFeatureMatrix(mi_matrix.reshape(data._ncol, data._ncol))
        

    def sampleCount(self):

        return len(self._sample_names)

    def sampleNames(self, data):

        return data._sample_names

    def featureCount(self):

        return len(self._feature_names)

    def featureNames(self, data):

        return data._feature_names

    def _solutions(self, mi_threshold = -float('inf'), causality_threshold = float('inf')):
        # filters[target][solution, ] is a vector of selected features
        # in a solution for a target; missing values denote removed features
        ## One question is why we need the string here?
        filters = pd.Series()
        target_indices = self._target_indices
        for target_index in self._target_indices:
            result_matrix = self._filters.loc[[target_index]]
            causality_dropped, _ = np.where(self._causality_list.loc[[target_index]] > causality_threshold)
            mi_dropped, _ = np.where(-.5 * np.log(1 - np.square(self._mi_matrix[:, target_index])) < mi_threshold)
             
            # Apply the Nan operation here
            for idx in set(list(causality_dropped) + list(mi_dropped)):
                i, j = idx % result_matrix.shape[0], idx // result_matrix.shape[0]
                result_matrix[i][j] = np.nan 
    
            pre_return_matrix = np.flip(result_matrix, axis = 0)
            filters = filters.append(pd.Series([pre_return_matrix]))
        
        filters.index = target_indices

        return filters

    def _scores(self):
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

        