import pandas as pd
import unittest
import requests as rq
import io 

from pymrmre import mrmr

class TestPymrmre(unittest.TestCase):

    def setUp(self):
        
        data_url = 'https://raw.githubusercontent.com/bhklab/PymRMRe/master/data/cgps_ge.csv'
        data = rq.get(data_url).content
        
        self.data = pd.read_csv(io.StringIO(data.decode('utf-8')), index_col = 0)
        

        self.feats = {
            "geneid_9761,NA,1,10" : [[48, 75, 84, 58, 14, 35, 7, 62, 52, 73]],
            "geneid_9761,NA,2,10" : [[75, 35, 84, 48, 7, 14, 62, 52, 73, 15],
                                [48, 75, 84, 58, 14, 35, 7, 62, 52, 73]],
            "geneid_9761,geneid_3310,2,10" : [[0, 75, 35, 48, 84, 7, 14, 52, 62, 73],
                                [0, 48, 75, 84, 58, 35, 14, 7, 52, 62]]
        }


    def test_mRMRe(self):
        for s in self.feats.keys():
            inputs = s.split(',')  

            targets = pd.DataFrame(self.data[inputs[0]])

            features = self.data.drop(inputs[0], axis=1)

            if inputs[1] == 'NA':
                fixed_features = []
            else:
                fixed_features = [inputs[1]]

            feats = mrmr.mrmr_ensemble(features = features, 
                                    targets = targets, 
                                    fixed_features = fixed_features,
                                    solution_count = int(inputs[2]), 
                                    solution_length = int(inputs[3]),
                                    return_index = True)
                
            assert self.feats[s] == feats.iloc[0]

R_TIME_RESULTS = [
    "lbp-3D-k_glszm_LargeAreaLowGrayLevelEmphasis",
    "logarithm_firstorder_Skewness",
    "lbp-2D_glszm_ZoneVariance",
    "lbp-3D-k_glszm_ZoneVariance",
    "logarithm_firstorder_Skewness",
    "lbp-2D_glszm_ZoneVariance",
    "original_glrlm_GrayLevelNonUniformity",
    "logarithm_glcm_Imc2",
    "wavelet-LHL_glcm_ClusterShade",
    "lbp-3D-k_glszm_LargeAreaEmphasis",
    "logarithm_firstorder_Skewness",
    "lbp-2D_glszm_ZoneVariance",
    "lbp-3D-k_glszm_LargeAreaHighGrayLevelEmphasis",
    "logarithm_firstorder_Skewness",
    "lbp-2D_glszm_ZoneVariance",
]

R_SURVIVAL_RESULTS = [
    "lbp-2D_firstorder_Median",
    "lbp-2D_firstorder_90Percentile",
    "lbp-2D_firstorder_10Percentile",
    "lbp-3D-m1_firstorder_Median",
    "lbp-2D_firstorder_90Percentile",
    "lbp-2D_firstorder_Median",
    "lbp-2D_firstorder_10Percentile",
    "lbp-3D-m1_firstorder_Median",
    "exponential_glcm_Imc1",
    "lbp-2D_firstorder_90Percentile",
    "lbp-2D_firstorder_10Percentile",
    "lbp-3D-m1_firstorder_Median",
    "lbp-2D_glszm_ZoneVariance",
    "lbp-2D_firstorder_90Percentile",
    "lbp-2D_firstorder_10Percentile",
]

R_SURVIVAL_RESULTS_ONE = [
    "lbp-2D_firstorder_Median",
    "lbp-3D-m1_firstorder_Median",
    "lbp-2D_firstorder_10Percentile",
    "lbp-2D_firstorder_90Percentile",
    "lbp-2D_glszm_ZoneVariance",
]

R_TIME_RESULTS_ONE = [
    "lbp-3D-k_glszm_LargeAreaLowGrayLevelEmphasis",
    "lbp-3D-k_glszm_ZoneVariance",
    "original_glrlm_GrayLevelNonUniformity",
    "lbp-3D-k_glszm_LargeAreaEmphasis",
    "lbp-3D-k_glszm_LargeAreaHighGrayLevelEmphasis",
]


class TestREquivalent(unittest.TestCase):
    def setUp(self):
        import os
        import pandas
        import numpy as np
        data_path = os.path.join(os.path.dirname(__file__), "../data")
        self.joined_features = pandas.read_csv(os.path.join(data_path, "r_data.csv")).astype(np.float64)
        self.features = self.joined_features.drop(columns=['event', 'time'])
        self.surv = self.joined_features[['event', 'time']]
        self.event = self.joined_features['event']
        self.time = self.joined_features['time']

    def test_mrmr_ensemble_survival_one(self):
        results = mrmr.mrmr_ensemble_survival(self.features, self.surv, solution_length=1, solution_count=5)
        results = results.iloc[0]
        results = [r for res in results for r in res]
        self.assertEqual(results, R_SURVIVAL_RESULTS_ONE)
        
    def test_mrmr_ensemble_one(self):
        results = mrmr.mrmr_ensemble(self.features, self.time.to_frame(), solution_length=1, solution_count=5)
        results = results.iloc[0]
        results = [r for res in results for r in res]
        self.assertEqual(results, R_TIME_RESULTS_ONE)

    def test_mrmr_ensemble_survival(self):
        import numpy as np
        results = mrmr.mrmr_ensemble_survival(self.features, self.surv, solution_length=3, solution_count=5)
        results = results.iloc[0]
        results = [r for res in results for r in res]
        self.assertEqual(results, R_SURVIVAL_RESULTS)
        
    def test_mrmr_ensemble(self):
        results = mrmr.mrmr_ensemble(self.features, self.time.to_frame(), solution_length=3, solution_count=5)
        results = results.iloc[0]
        results = [r for res in results for r in res]
        self.assertEqual(results, R_TIME_RESULTS)


if __name__ == '__main__':
    unittest.main()

