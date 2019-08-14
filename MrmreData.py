import numpy as np 
import pandas as pd 
import math
import expt
import constants
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

class MrmreData:
  
    def __init__(self,
                 data : pd.DataFrame = None,
                 strata : pd.Series = None,
                 weights : pd.Series = None,
                 priors : np.array = None):

        # Import the necessary r libraries needed for mRMRe
        self.survival = importr('survival')

        ## Declare the private or protected variables here
        self._data           = pd.DataFrame()
        self._feature_types  = pd.Series()
        self._strata         = pd.Series()
        self._weights        = pd.Series()
        self._priors         = np.array()
        self._sample_names   = list()
        self._feature_names  = list()
        self._estimator_map = {'pearson'  : constants.ESTIMATOR.PEARSON, 
                               'spearman' : constants.ESTIMATOR.SPEARMAN, 
                               'kendall'  : constants.ESTIMATOR.KENDALL,
                               'frequency': constants.ESTIMATOR.FREQUENCY}

        self._features_map = {'continuous': constants.FEATURE.CONTINUOUS,
                              'discrete'  : constants.FEATURE.DISCRETE,
                              'time'      : constants.FEATURE.TIME,
                              'event'     : constants.FEATURE.EVENT}

        if not isinstance(data, pd.DataFrame):
            raise Exception('Data must be of type dataframe')
        if data.shape[1] > (math.sqrt(2^31) - 1):
            raise Exception("Too many features, the number of features should be <= 46340")
 
        # Define the feature types of the data
        for _, col in data.iteritems():
            # Firstly check whether the feature is survival data (depends on the column names)
            if col.name in ['time', 'event']:
                self._feature_types.append(pd.Series([self._feature_map[col.name]]))
                continue
            
            # If not, check the feature is numeric data or categorical data (ordered-factor)
            if np.issubdtype(col.dtype, np.number):
                self._feature_types.append(pd.Series([self._feature_map['continuous']]))
            elif col.dtype.name == 'category':
                self._feature_types.append(pd.Series([self._feature_map['discrete']]))
            else:
                raise Exception("Wrong labels")

        self._sample_names = list(data.index.values)
        self._feature_names = list(data.column.values)

        # Build the mRMR data
        if self._feature_types.sum() == 0:
            self._data = data
        else:
            for i, _feature_type in self._feature_types.iteritems():
                if _feature_type == self._feature_map['continuous']:
                    # With the column name
                    _feature = pd.to_numeric(data.iloc[:, i])
                elif _feature_type in (self._feature_map['time'], self._feature_map['event']):
                    _feature = data.iloc[:, i]
                else:
                    # Why minus one? Is the indexing problems between R and C++?
                    _feature = data.iloc[:, i].astype(int) - 1

                self._data[_feature.name] = _feature
        
        # Naming the new dataframe (with column names)
        ## Already done since the dataframe is composed of pandas series (with names)

        # Sample Stratum processing
        self._strata = strata if strata else pd.Series(np.zeros(data.shape[0]))

        # Sample Weight processing
        self._weights = weights if weights else pd.Series(np.ones(data.shape[0]))

        # Sample Feature Matrix Processing
        self._priors = priors

        # No explictly return in __init__ function

    def featureData(self):
        '''
        ## Apply one Surv class here to build Surv object
        :return: one dataframe with feature data
        '''
        ## Still need to figure out what it want to return
        ## Still what about the survival data? 
        feature_data = pd.DataFrame()
        for i, _feature_type in self._feature_types.iteritems():
            if _feature_type == self._feature_map['continuous']:
                _feature = self._data.iloc[:, i]
            elif _feature_type == self._feature_map['discrete']:
                _feature = self._data.iloc[:, i] + 1
            else:
                continue
            feature_data[_feature.name] = _feature
        
        feature_data = feature_data.dropna()
        # Build the censored data (need to check again how filter used here)
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

        _data = self.featureData().iloc[row_indices, col_indices]
        _strata = pd.factorize(self.sampleStrata().iloc[row_indices], sort = True)
        _weights = self.sampleWeights().iloc[row_indices]
        _priors = self.priors()[col_indices, col_indices] if self.priors() else None

        return self.__init__(_data, _strata, _weights, _priors)


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
        return len(self._feature_names)
    
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
            _strata = self._strata
            _strata.index = list(self._data.index.values)
            return _strata

        else:
            if len(value) != self._data.shape[0]:
                raise Exception('Data and strata must contain the same number of samples')
            elif value.dtype.name != 'category':
                raise Exception('Strata must be provided as factors')
            elif value.isnull().any().any():
                raise Exception('Cannot have missing value in strata')
            # Why we need to return self variable in this case?
            # Do we need to minus one here?
            self._strata = value.astype(int)

    ## SampleWeights
    def sampleWeights(self, value : pd.Series = None):
        '''
        :param value:
        :return:
        '''
        if not value:
            _weights = self._weights
            _weights.index = list(self._data.index.values)
            return _weights
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
        _mi_matrix = np.empty([self._data.shape[1], self._data.shape[1]])
        ######################
        ## Need to convert all 2d arrays to 1d ??
        ######################
        expt.export_mim(self._data.values.flatten(), 
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
                        _mi_matrix.flatten())
        
        _mi_matrix = self._compressFeatureMatrix(_mi_matrix)
        
        return _mi_matrix

    ## Helper function to build censored data
    def _survBuild(self, row):
        return np.array([row['time'], row['event']])

    ## Helper function to expand FeatureMatrix
    # It seems like functions about feature matrix return array, but the functions about 
    # feature indices return pandas series ? Should still be array

    def _expandFeatureMatrix(self, matrix):
        expanded_matrix = np.array()
        adaptor = self._feature_types.index[self._feature_types != 3].tolist()
        for i in range(len(adaptor)):
            col = np.array()
            for j in range(len(adaptor)): 
                # Row binding 
                item = matrix[adaptor[j]][adaptor[i]]
                if self._feature_types[adaptor[j]] == 2:
                    col = np.vstack((col, item))
                    col = np.vstack((col, item))
                else:
                    col = np.vstack((col, item))
            # Column binding
            if self._feature_types[adaptor[i]] == 2:
                expanded_matrix = np.hstack((expanded_matrix, col))
                expanded_matrix = np.hstack((expanded_matrix, col))
            else:
                expanded_matrix = np.hstack((expanded_matrix, col))

        return expanded_matrix
        
    ## Helper function to compress FeatureMatrix
    def _compressFeatureMatrix(self, matrix):
        
        adaptor = self._feature_types.index[self._feature_types != 3].tolist()

        return matrix[adaptor, adaptor]

    ## expandFeatureIndices
    def _expandFeatureIndices(self, indices):
        indices = list(indices)
        adaptor = self._feature_types.index[self._feature_types == 3].tolist()
        if len(adaptor) > 0 and (indices >= adaptor).any():
            for i in range(len(indices)):
                for j in range(len(adaptor)):
                    # 0/1 Indexing problem? Why plus one here?
                    ## Does not matter
                    indices[i] += (indices[i] >= (adaptor[j] - j + 1))

        return np.array(indices)

    ## compressFeatureIndices
    def _compressFeatureIndices(self, indices):
        indices = list(indices)
        _adaptor = self._feature_types.index[self._feature_types == 3].tolist()
        # It's correct here
        if len(_adaptor) > 0:
            for i in range(len(indices)):
                for j in range(len(_adaptor)):
                    indices[i] -= (indices[i] >= _adaptor[j])

        return np.array(indices)

    def _scores(self, solutions):
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




        return 



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
    
