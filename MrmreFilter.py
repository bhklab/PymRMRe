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

        if any(x < 1 for x in target_indices) or any(x > data.featureCount() for x in target_indices):
            raise Exception('target indices must only contain values ranging from 1 to the number of features in data')
        
        ## Level processing

        if not levels:
            raise Exception('levels must be provided')

        self._target_indices = target_indices.astype(int)
        self._levels = levels.astype(int)

        # The index 0/1 problems?
        target_indices = data.expandFeatureIndices(target_indices).astype(int) - 1

        ## Filter; Mutual Information and Causality Matrix

        mi_matrix = np.zeros((data._nrow, data._ncol))

        if method == 'exhaustive':

            ## Level processing
            if np.prod(levels) - 1 > comb(data._featureCount - 1, len(levels)):
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


        ## After building the result, result is the data structure of Result defind in exports.h
        ## The returned filter is object of list
        self._filters = []
        for i in range(len(result.solutions):
            sol_matrix = (data.compressFeatureIndices(result.solutions[i] + 1)).reshape(len(self._levels), np.prod(self._levels))
            self._filter.append(sol_matrix)
        self._causality_list = list(result.casuality)

        self._scores = [i.reshape(len(self._levels), np.prod(self._levels)) for i in result.scores]
        _, cols_unique = np.unique(data.compressFeatureIndices(list(range(data._ncol))))
        # Do we need to store the names of targets?
        self._causality_list = [causality[cols_unique] for causality in self._causality_list]
        self._mi_matrix = data.compressFeatureMatrix(mi_matrix.reshape(data._ncol, data._ncol))

        # Ignore the feature names / sample names

    

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
        ## One question is why we need the string here?
        filters = []
        for target_index in self._target_indices:
            res


            result_matrix = self._filters[target_index]
            ### numpy where return a turple, the first element is the indices array
            causality_dropped = np.where(np.array(self._casuality_list[str(target_index)] > causality_threshold))[0]
            mi_dropped = np.where(np.array(-.5 * np.log(1 - np.square(self._mi_matrix[:,target_index])) < mi_threshold))[0]
            # Need to apply the Nan operation here?
            pre_return_matrix = np.flip(result_matrix, axis = 1)
        filters.append(pre_return_matrix)

        return filters

    def scores(self):
        mi_matrix = self.mim()
        # No need to use the string style of targets, it is for the names of columns in R
        targets = self._target_indices

        scores = []
        for target in targets:



        return

    def mim(self, method):
        ## The method should be within mi or cor
        return self._mi_matrix

    def target(self):
        return self._target_indices

        