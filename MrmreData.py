import numpy as np 
import pandas as pd 
import math
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

class MrmreData:
    
    def __init__(self,
                 data : pd.DataFrame = None,
                 strata : np.array = None,
                 weights : np.array = None,
                 priors : np.array = None):

        # Import the necessary r libraries needed for mRMRe
        self.survival = importr('survival')

        self._data = np.array()
        ## The case when features are too many
        if data.shape[1] > (math.sqrt(2^31) - 1):
            raise Exception("Too many features, the number of features should be <= 46340")
 
        # Check the data type of features
        self._feature_types = []
        for _, col in data.iteritems():
            # Firstly check whether the feature is survival data
            if col.name == 'time':
                self._feature_types.append(2)
                continue
            elif col.name == 'event':
                self._feature_types.append(3)
                continue
            
            # If not, check the feature is numeric data or categorical data (ordered-factor)
            if np.issubdtype(col.dtype, np.number):
                self._feature_types.append(0)
            elif col.dtype.name == 'category':
                self._feature_types.appebd(1)

        ## Build the mRMR data
        # All features are numeric type, convert the dataframe to matrix directly
        if sum(feature_types) == 0:
            self._data = data.values
        # The feature data include categorical and survival data
        ## Bo: There may exist some problem here (Surv). Why we do not need to create Surv here?
        # Finally the self._data is matrix type, not dataframe
        else:
            for i in range(len(feature_types)):
                if feature_types[i] == 0:
                    feature = pd.to_numeric(data.iloc[i])
                elif feature_types[i] == 2 or feature_types[i] == 3:
                    feature = data.iloc[i]
                else:
                    feature = data.iloc[i].astype(int)

                self._data = np.hstack(self._data, feature)

        # Sample Stratum Processing
        if not strata:
            self._strata = np.zeros(data.shape[0])
        else:
            self._strata = strata

        # Sample Weight Processing
        if not weights:
            self._weights = np.ones(data.shape[0])
        else:
            self._weights = weights

        # Prior Feature Matrix Processing
        if priors:
            self._priors = priors

        # No explictly return in __init__ function

    def featureData(self):
        # Apply Surv here
        # return dataframe here?
        for i in range(self._data.shape[1]):
            if self._feature_types[i] == 0:
        return

    def subsetData(self, 
                   row_indices : list = None, 
                   col_indices : list = None):
        
        if not row_indices and not col_indices:
            return self._data 
        
        if not row_indices:
            row_indices = list(range(self.sampleCount))
        if not col_indices:
            col_indices = list(range(self.featureCount))

        data = self.featureData.iloc[row_indices, col_indices]
        strata = 
        

    ## SampleCount
    def sampleCount(self):
        return self._data.shape[0]

    ## SampleName
    def sampleNames(self):
        return 
    
    ## featureCount
    def featureCount(self):
        return len(self._feature_names)
    
    ## featureNames
    def featureCount(self):
        return 
    
    ## sampleStrata
    def sampleStrata(self, value : pd.Series = None):
        if not value:
            return self._strata
        if len(value) != self._data.shape[0]:
            raise Exception('Data and strata must contain the same number of samples')
        elif value.dtype.name == 'category':
            raise Exception('Strata must be provided as factors')
        else:
            self._strata = value.astype(int) - 1

    ## SampleWeights
    def sampleWeights(self, value : pd.Series = None):
        if not value:
            return self._weights
        if len(value) != self._data.shape[0]:
            raise Exception('Data and weight must contain the same number of samples')
        elif value.isnull.values.any():
            raise Exception('cannot have missing values in weights')
        else:
            self._weights = value.astype(float)

    ## Priors
    def priors(self, value):
        if value.shape[0] != self._data.shape[0] or value.shape[1] != self._data.shape[1]:
            raise Exception('Priors matrix must be a symmetric matrix containing as many features as data')
        
        self._priors = self.expandFeatureMatrix(value)
    
    ## Mutual information matrix
    def mim(self, prior_weight = 0, continuous_estimator, outX = True, bootstrap_count):
        if continuous_estimator not in ['pearson', 'spearman', 'kendall', 'frequency']:
            raise Exception('The continuous estimator should be one of pearson, spearman, kendall and frequency')
        if len(self._priors) != 0:
            if prior_weight < 0 or prior_weight > 1:
                raise Exception('prior weight must be a value ranging from 0 to 1')
        else:
            prior_weight = 0
        '''
        call the cpp function
        '''
        return

    ## expandFeatureMatrix
    def expandFeatureMatrix(self, matrix):
        adaptor = np.where(np.array(self._feature_types) != 3)[0]
        # ???????
        matrix = matrix.tolist()
        for i in range(len(adaptor)):
            for j in range(len(adaptor)):
                #item = matrix[j][i]
                if self._feature_types[adaptor[j]] == 2:
                     



        return 
        
    ## compressFeatureMatrix
    def compressFeatureMatrix(self, matrix):
        # DONE
        adaptor = np.where(np.array(self._feature_types != 3))[0]

        return matrix[adaptor, adaptor]

    ## expandFeatureIndices
    def expandFeatureIndices(self, indices):
        adaptor = np.where(np.array(self._feature_types == 3))[0]
        if len(adaptor) > 0 and (indices >= adaptor).any():
            for i in range(len(indices)):
                temp = 0
                for j in range(len(adaptor)):
                    # 0/1 Indexing problem? Why plus one here?
                    if indices[i] >= adaptor[j] - j + 1:
                        temp += 1 
                indices[i] += temp
                temp = 0
        
        return indices

    ## compressFeatureIndices
    def conpressFeatureIndices(self, indices):
        adaptor = np.where(np.array(self._feature_types == 3))[0]
        if len(adaptor) > 0:
            for i in range(len(indices)):
                temp = 0
                for j in range(len(adaptor)):
                    if indices[i] >= adaptor[j]:
                        temp += 1
                indices[i] -= temp
                temp = 0
        return indices

    def scores(self, solutions):
        mi_matrix = self.mim()
        # ???


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
    
