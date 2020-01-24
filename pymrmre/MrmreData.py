import numpy as np 
import pandas as pd 
import math
from .constants import *
from expt import *

class MrmreData:
  
    def __init__(self,
                 data : pd.DataFrame = None,
                 feature_types : list = [],
                 strata : pd.Series = None,
                 weights : pd.Series = None,
                 priors : np.array = np.array([])):


        ## Declare the private or protected variables here
        self._data           = pd.DataFrame()
        self._feature_types  = pd.Series()
        self._strata         = pd.Series()
        self._weights        = pd.Series()
        self._priors         = np.array([])
        self._sample_names   = list()
        self._feature_names  = list()
        

        self._sample_names = list(data.index.values)
        self._feature_names = list(data.columns.values)

        if not isinstance(data, pd.DataFrame):
            raise Exception('Data must be of type dataframe')
        if data.shape[1] > (math.sqrt(2 ** 31 - 1)):
            raise Exception("Too many features, the number of features should be <= 46340")
 
        # Build the feature types

        self._feature_types = pd.Series(feature_types)
        self._feature_types.index = list(range(data.shape[1]))
       
        
        ## Build the mRMR data
        
        if self._feature_types.sum() == 0:
            self._data = data
        else:
            for i, col in enumerate(data):
                if self._feature_types[i] == 1:     # Factor variables
                    self._data[col] = data.loc[:, col].astype(int) - 1
                else:  
                    self._data[col] = data.loc[:, col]
        

        # Sample Stratum processing
        self._strata = strata if strata else pd.Series(np.zeros(data.shape[0]))
        self._strata.index = self._sample_names

        # Sample Weight processing
        self._weights = weights if weights else pd.Series(np.ones(data.shape[0]))
        self._weights.index = self._sample_names

        # Sample Feature Matrix Processing
        self._priors = priors

    ## No need for the featureData() function in Python package
    ## 
    def featureData(self):
        '''
        ## Apply one Surv class here to build Surv object
        :return: one dataframe with feature data
        '''
        ## Still need to figure out what it want to return
        ## Still what about the survival data? 
        feature_data = pd.DataFrame()
        for i, feature_type in self._feature_types.iteritems():
            if feature_type == self._feature_map['continuous']:
                feature = self._data.iloc[:, i]
            elif feature_type == self._feature_map['discrete']:
                feature = self._data.iloc[:, i] + 1
            else:
                continue
            feature_data[feature.name] = _feature
        
        feature_data = feature_data.dropna()
        # Build the censored data (need to check again how filter used here)
        # Here the time and event columns are not included 
        feature_data['censored'] = feature_data.apply(self._survBuild, axis = 1)

        return feature_data

    def subsetData(self,
                   row_indices : list = None, 
                   col_indices : list = None):
        '''
        :param row_indices:
        :param col_indices:
        :return: A subset of mRMR data
        '''
        if not row_indices and not col_indices:
            return self._data
        
        if not row_indices:
            row_indices = list(range(self.sampleCount()))
        if not col_indices:
            col_indices = list(range(self.featureCount()))

        data = self.featureData().iloc[row_indices, col_indices]
        strata = pd.factorize(self.sampleStrata().iloc[row_indices], sort = True)
        weights = self.sampleWeights().iloc[row_indices]
        priors = self._priors()[col_indices, col_indices] if self._priors() else None

        return self.__init__(data, strata, weights, priors)


    ## SampleCount
    def sampleCount(self):
        '''
        :return:
        '''
        return self._data.shape[0]

    ## SampleName
    def sampleNames(self):
        '''
        :return:
        '''
        return self._sample_names
    
    ## featureCount
    def featureCount(self):
        '''
        :return:
        '''
        return self._data.shape[1]
    
    ## featureNames
    def featureNames(self):
        '''
        :return:
        '''
        return self._feature_names
    
    ## sampleStrata
    def sampleStrata(self, value : pd.Series = None):
        '''
        :param value:
        :return:
        '''
        if not value:
            strata = self._strata
            strata.index = list(self._data.index.values)
            return strata

        else:
            if len(value) != self._data.shape[0]:
                raise Exception('Data and strata must contain the same number of samples')
            elif value.dtype.name != 'category':
                raise Exception('Strata must be provided as factors')
            elif value.isnull().any().any():
                raise Exception('Cannot have missing value in strata')
            self._strata = value.astype(int)

    ## SampleWeights
    def sampleWeights(self, value : pd.Series = None):
        '''
        :param value:
        :return:
        '''
        if not value:
            weights = self._weights
            weights.index = list(self._data.index.values)
            return weights
        else:
            if value.size != self._data.shape[0]:
                raise Exception('Data and weight must contain the same number of samples')
            elif value.isnull.any().any():
                raise Exception('cannot have missing values in weights')
            
            self._weights = value.astype(float)        

    ## Priors
    def _priors(self, value):
        '''
        :param value: A numpy matrix
        :return: 
        '''
        if not value:
            return self._compressFeatureMatrix(self._priors) if self._priors else None
        else:
            if value.shape[0] != self._data.shape[0] or value.shape[1] != self._data.shape[1]:
                raise Exception('Priors matrix must be a symmetric matrix containing as many features as data')
            self._priors = self._expandFeatureMatrix(value)

    ## Mutual information matrix
    def _mim(self, prior_weight = 0, continuous_estimator = 'pearson', outX = True, bootstrap_count = 1):
        if continuous_estimator not in ['pearson', 'spearman', 'kendall', 'frequency']:
            raise Exception('The continuous estimator should be one of pearson, spearman, kendall and frequency')
        if self._priors:
            if not prior_weight:
                raise Exception('Prior weight must be provided if there are priors')
            elif prior_weight < 0 or prior_weight > 1:
                raise Exception('Prior weight must be a value ranging from 0 to 1')
        else:
            prior_weight = 0
        
        '''
        call the export_mim cpp function
        '''
        mi_matrix = np.empty([self._data.shape[1], self._data.shape[1]])
        export_mim(self._data.values.flatten(), 
                        self._priors.flatten(), 
                        prior_weight, 
                        self._strata.values.astype(int), 
                        self._weights.values, 
                        self._feature_types.values, 
                        self._data.shape[0],
                        self._data.shape[1], 
                        len(self._strata.unique()), 
                        self._estimator_map[continuous_estimator], 
                        int(outX == true), 
                        bootstrap_count, 
                        mi_matrix.flatten())
        
        mi_matrix = self._compressFeatureMatrix(mi_matrix)
        
        return mi_matrix


    ## Helper function to expand FeatureMatrix

    def _expandFeatureMatrix(self, matrix):
        expanded_matrix, adaptor = [], []
        # Compute the adaptor
        i = 0
        for _, item in self._feature_types.iteritems():
            if item != 3:
                adaptor.append(i)
            i += 1

        for i in range(len(adaptor)):
            col = []
            for j in range(len(adaptor)): 
            # Row binding 
                item = matrix[adaptor[j]][adaptor[i]]
                if self._feature_types.iloc[adaptor[j]] == 2:
                    col.append(item)     # Extra prior for Surv object
                col.append(item)
            # Column binding
            if self._feature_types.iloc[adaptor[i]] == 2:
                expanded_matrix.append(col)
            expanded_matrix.append(col)

        return np.array(expanded_matrix).T
        
    ## Helper function to compress FeatureMatrix
    def _compressFeatureMatrix(self, matrix):
       # Compute the adaptor
        i, adaptor = 0, []
        for _, item in self._feature_types.iteritems():
            if item != 3:
                adaptor.append(i)
            i += 1
        
        return matrix[adaptor][:, adaptor]

    ## expandFeatureIndices
    def _expandFeatureIndices(self, indices):
        indices = list(indices)
        # Compute the adaptor
        i, adaptor = 1, []
        for _, item in self._feature_types.iteritems():
            if item == 3:
                adaptor.append(i)
            i += 1
            
        if len(adaptor) > 0:
            for i, index in enumerate(indices):
                for j in range(len(adaptor)):
                    indices[i] += (index >= (adaptor[j] - j))
        
        return np.array(indices)

    ## compressFeatureIndices
    def _compressFeatureIndices(self, indices):
        indices = list(indices)
        # Compute the adaptor
        i, adaptor = 1, []
        for _, item in self._feature_types.iteritems():
            if item == 3:
                adaptor.append(i)
            i += 1

        if len(adaptor) > 0:
            for i, index in enumerate(indices):
                indices[i] -= len([j for j in adaptor if index >= j])

        return np.array(indices)

    def scores(self, solutions):
        '''
        Better develop that after the finish of mRMR.Filter class
        '''
        mi_matrix = self._mim()
        target_indices = solutions.index.values.tolist()
        scores = pd.Series()

        for i, target in enumerate(target_indices):
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
            
            scores.append(sub_score_target)
        
        scores.reindex(target_indices)
        
        return scores







    '''

    indices <- sapply(indices, 
    function(i) {
        i + sum(
            sapply(1:length(adaptor), 
                   function(j) i >= (adaptor[[j]] - j + 1)
                )
            )
    })
    '''
    
