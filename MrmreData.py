import numpy as np 
import pandas as pd 
import math
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

class MrmreData:

    # Data type constants
    NUMERIC  = 0
    FACTOR   = 1
    TIME     = 2
    EVENT    = 3

    def __init__(self,
                 data : pd.DataFrame = None,
                 strata : pd.Series = None,
                 weights : pd.Series = None,
                 priors : np.array = None):

        # Import the necessary r libraries needed for mRMRe
        self.survival = importr('survival')

        ## Declare the private or protected variables here
        self.__data           = pd.DataFrame()
        self.__feature_types  = pd.Series()
        self.__strata         = pd.Series()
        self.__weights        = pd.Series()
        self.__priors         = np.array()
        self.__sample_names   = list()
        self.__feature_names  = list()

        if not isinstance(data, pd.DataFrame):
            raise Exception('Data must be of type dataframe')
        if data.shape[1] > (math.sqrt(2^31) - 1):
            raise Exception("Too many features, the number of features should be <= 46340")
 
        # Define the feature types of the data
        for _, col in data.iteritems():
            # Firstly check whether the feature is survival data
            if col.name == 'time':
                self.__feature_types.append(pd.Series([self.TIME]))
                continue
            elif col.name == 'event':
                self.__feature_types.append(pd.Series([self.EVENT]))
                continue
            # If not, check the feature is numeric data or categorical data (ordered-factor)
            if np.issubdtype(col.dtype, np.number):
                self.__feature_types.append(pd.Series([self.NUMERIC]))
            elif col.dtype.name == 'category':
                self.__feature_types.append(pd.Series([self.FACTOR]))
            else:
                raise Exception("Wrong labels")

        self.__sample_names = list(data.index.values)
        self.__feature_names = list(data.column.values)

        # Build the mRMR data
        if self.__feature_types.sum() == 0:
            self.__data = data
        else:
            for i, __feature_type in self.__feature_types.iteritems():
                if __feature_type == self.NUMERIC:
                    # With the column name
                    __feature = pd.to_numeric(data.iloc[:, i])
                elif __feature_type in (self.TIME, self.EVENT):
                    __feature = data.iloc[:, i]
                else:
                    # Why minus one? Is the indexing problems between R and C++?
                    __feature = data.iloc[:, i].astype(int) - 1

                self.__data[__feature.name] = __feature

        # Sample Stratum processing
        self.__strata = strata if strata else pd.Series(np.zeros(data.shape[0]))

        # Sample Weight processing
        self.__weights = weights if weights else pd.Series(np.ones(data.shape[0]))

        # Sample Feature Matrix Processing
        self.__priors = priors

        # No explictly return in __init__ function

    def featureData(self):
        '''
        ## Apply one Surv class here to build Surv object
        :return: one dataframe with feature data
        '''
        ## Still need to figure out what it want to return
        for i in range(self._data.shape[1]):
            if self._feature_types[i] == 0:



        return

    def subsetData(self,
                   row_indices : list = None, 
                   col_indices : list = None):
        '''
        :param row_indices:
        :param col_indices:
        :return: A subset of mRMR data
        '''
        if not row_indices and not col_indices:
            return self.__data
        
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
        return self.__data.shape[0]

    ## SampleName
    def sampleNames(self):
        '''
        :return:
        '''
        return self.__sample_names
    
    ## featureCount
    def featureCount(self):
        '''
        :return:
        '''
        return len(self.__feature_names)
    
    ## featureNames
    def featureNames(self):
        '''
        :return:
        '''
        return self.__feature_names
    
    ## sampleStrata
    def sampleStrata(self, value : pd.Series = None):
        '''
        :param value:
        :return:
        '''
        if not value:
            _strata = self.__strata
            _strata.index = list(self.__data.index.values)
            return _strata

        else:
            if len(value) != self.__data.shape[0]:
                raise Exception('Data and strata must contain the same number of samples')
            elif value.dtype.name != 'category':
                raise Exception('Strata must be provided as factors')
            elif value.isnull().any().any():
                raise Exception('Cannot have missing value in strata')
            # Why we need to return self variable in this case?
            # Do we need to minus one here?
            self.__strata = value.astype(int)

    ## SampleWeights
    def sampleWeights(self, value : pd.Series = None):
        '''
        :param value:
        :return:
        '''
        if not value:
            _weights = self.__weights
            _weigths.index = list(self.__data.index.values)
            return _weights
        else:
            if value.size != self.__data.shape[0]:
                raise Exception('Data and weight must contain the same number of samples')
            elif value.isnull.any().any():
                raise Exception('cannot have missing values in weights')
            
            self.__weights = value.astype(float)        

    ## Priors
    def priors(self, value):
        '''
        :param value: A numpy matrix
        :return: 
        '''
        if not value:
            return self.__compressFeatureMatrix(self.__priors) if self.__priors else None
        else:
            if value.shape[0] != self._data.shape[0] or value.shape[1] != self._data.shape[1]:
                raise Exception('Priors matrix must be a symmetric matrix containing as many features as data')
            self.__priors = self.__expandFeatureMatrix(value)

    ## Mutual information matrix
    def mim(self, prior_weight = 0, continuous_estimator = None, outX = True, bootstrap_count = 1):
        if continuous_estimator not in ['pearson', 'spearman', 'kendall', 'frequency']:
            raise Exception('The continuous estimator should be one of pearson, spearman, kendall and frequency')
        if self.__priors:
            if not prior_weight:
                raise Exception('Prior weight must be provided if there are priors')
            elif prior_weight < 0 or prior_weight > 1:
                raise Exception('Prior weight must be a value ranging from 0 to 1')
        else:
            prior_weight = 0
        
        _mi_matrix = 
        '''
        call the cpp function
        '''
        return _mi_matrix

    ## expandFeatureMatrix
    def __expandFeatureMatrix(self, matrix):
        _expanded_matrix = np.array()
        _adaptor = self.__feature_types.index[self.__feature_types != 3].tolist()
        for i in range(len(_adaptor)):
            col = np.array()
            for j in range(len(_adaptor)):
                # item should be relevant to adaptor?
                item = matrix[_adaptor[j]][_adaptor[i]]
                if self.__feature_types[_adaptor[j]] == 2:
                    col = col.vstack([item], [item])
                else:
                    col = col.vstack([item])
            if self.__feature_types[_adaptor[i]] == 2:
                expanded_matrix.hstack(col)
                expanded_matrix.hstack(col)
            else:
                expanded_matrix.hstack(col)

        return expanded_matrix
        
    ## compressFeatureMatrix
    def __compressFeatureMatrix(self, matrix):
        
        _adaptor = self.__feature_types.index[self.__feature_types != 3].tolist()

        return matrix[_adaptor, _adaptor]

    ## expandFeatureIndices
    def __expandFeatureIndices(self, indices):
        # Compare the list and array? 
        _adaptor = self.__feature_types.index[self.__feature_types == 3].tolist()
        if len(_adaptor) > 0 and (indices >= _adaptor).any():
            for i in range(len(indices)):
                for j in range(len(_adaptor)):
                    # 0/1 Indexing problem? Why plus one here?
                    indices[i] += (indices[i] >= (_adaptor[j] - j + 1))

        return indices

    ## compressFeatureIndices
    def __compressFeatureIndices(self, indices):
        indices = np.array(indices)
        _adaptor = self.__feature_types.index[self.__feature_types == 3].tolist()
        # It's correct here
        if len(_adaptor) > 0:
            for i in range(len(indices)):
                for j in range(len(_adaptor)):
                    indices[i] -= (indices[i] >= _adaptor[j])

        return indices

    def scores(self, solutions):
        '''
        Better develop that after the finish of mRMR.Filter class
        '''
        mi_matrix = self.mim()
        targets = 




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
    
