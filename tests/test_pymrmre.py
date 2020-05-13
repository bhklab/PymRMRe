import pandas as pd
import unittest
import requests as rq
import io 

from pymrmre import mrmr

class TestPymrmre(unittest.TestCase):

    def setUp(self):
        
        data_url = 'https://raw.githubusercontent.com/bhklab/PymRMRe/master/data/cgps_ge.csv?token=AF7WH4Y2LWQ7W3PG5KQMWJ2542P66'
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

if __name__ == '__main__':
    unittest.main()

