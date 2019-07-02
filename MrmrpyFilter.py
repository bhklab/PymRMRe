import numpy as np
import pandas as pd
import math
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

pandas2ri.activate()

class MrmrpyData:

    def __init__(self,
                 data : pd.DataFrame):

        # Import the necessary r libraries needed for mRMRe
        self.survival = importr('survival')

        self._data = np.array()
        ## The case when features are too many
        if data.shape[1] > (math.sqrt(2^31) - 1):
            print("Too many features")
            return

        # Check the data type of features
        feature_types = list()
        for _ , cols in data.iteritems():
            # The feature is numeric data
            if np.issubdtype(cols.dtype, np.number):
                feature_types.append(0)

            # The feature is categorical data (Order-factor in R)
            elif cols.dtype.name == 'category':
                feature_types.append(1)

            # The feature is survival data


        # All features are numeric type, convert the dataframe to matrix directly
        if sum(feature_types) == 0:
            self._data = data.values

        # The feature data relevant to categorical and survival data
        else:


    def featureData(self):










    def surv(self, row):
        as_numeric = robjects.r['as.numeric']

        return self.survival.Surv(time=as_numeric(row['time']), event=as_numeric(row['event']))



