import numpy as np
import pandas as pd
from MrmreData import *
from MrmreFilter import *
from constants import *


if __name__ == "__main__":
    clinical_info = pd.read_csv('data/clinical_info.csv')
    features = pd.read_csv('data/OPC1_radiomic_features.csv')
    clinical = clinical_info.copy()
    clinical.set_index('id', inplace=True)
    features = pd.read_csv('data/OPC1_radiomic_features.csv')
    idx = features.columns.tolist()[1:]
    feature_names = features['Unnamed: 0'].values.tolist()
    features_trans = features.T.drop(['Unnamed: 0'])
    features_trans.columns = feature_names
    features_merged = features_trans.merge(clinical[['event', 'time']], how='inner', left_index=True, right_index=True)
    df = features_merged

    df_mrmr = MrmreData(data=df)
    print(df_mrmr._data.shape)

