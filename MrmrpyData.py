import numpy as np
import pandas as pd
import math
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

pandas2ri.activate()

class MrmrpyData:

    def __init__(self,
                 data : pd.DataFrame = None,
                 strata : np.array = None,
                 weights : np.array = None,
                 priors : np.array = None
                 ):

        # Import the necessary r libraries needed for mRMRe
        self.survival = importr('survival')

        self._data = np.array()
        ## The case when features are too many
        if data.shape[1] > (math.sqrt(2^31) - 1):
            print("Too many features, the number of features should be <= 46340")
            return

        # Check the data type of features
        feature_types = list()
        for _ , cols in data.iteritems():

            # Firstly, check whether the feature is survival data
            if cols.name == "time":
                feature_types.append(2)
                continue
            elif cols.name == "event":
                feature_types.append(3)
                continue

            # If not, check the feature is numeric data or categorical data (ordered-factor)
            if np.issubdtype(cols.dtype, np.number):
                feature_types.append(0)
            elif cols.dtype.name == 'category':
                feature_types.append(1)

        ## Build the mRMR data
        # All features are numeric type, convert the dataframe to matrix directly
        if sum(feature_types) == 0:
            self._data = data.values
        # The feature data include categorical and survival data
        ## Bo: There may exist some problem here
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

# The primary problem is that do we need to combine survival data as one column?
    def featureData(self):
        #TBD
        return

    def subsetData(self, rows = None, cols = None):

        if not rows and not cols:
            return self._data
        if not rows:
            rows = list(range(self.sample_count()))
        if not cols:

        return

    def sampleCount(self):

        return self._data.shape[0]

    def sampleNames(self):
        #TBD
        return

    def featureCount(self):
        #TBD
        return

    def featureNames(self):
        #TBD
        return

    def sampleStrata(self):
        #TBD
        return

    def sampleWeights(self):
        #TBD
        return

    def priors(self):
        #TBD
        return

    def mim(self):
        #TBD
        return

    def expandFeatureMatrix(self, matrix):
        #TBD
        return

    def compressFeatureMatrix(self, matrix):
        #TBD
        return

    def expandFeatureIndices(self, indices):
        #TBD
        return

    def compressFeatureIndices(self, indices):
        #TBD
        return

    def scores(self, solutions):
        #TBD
        return






    def surv(self, row):
        as_numeric = robjects.r['as.numeric']

        return self.survival.Surv(time=as_numeric(row['time']), event=as_numeric(row['event']))



