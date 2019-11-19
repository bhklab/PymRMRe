#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os

# os.environ["RHOME"] = "/Library/Frameworks/R.framework/Resources"
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

pandas2ri.activate()

#matrix = importr('Matrix')
#survival = importr('survival')
#igraph = importr('igraph')
utils = importr('utils')
utils.install_packages('mRMRe')
mrmre = importr('mRMRe')
from sklearn.feature_selection import VarianceThreshold


def variance_threshold_selector(data, threshold=0.01):
    selector = VarianceThreshold(threshold)
    selector.fit_transform(data)
    return data


def mrmr_selection(features: pd.DataFrame,
                   target_df: pd.DataFrame,
                   solution_count: int,
                   feature_count: int,
                   survial: bool = False):
    '''
     This function use mrmr feature selection to select specific number of features

     :param survial:
     :param features      : The radiomics features or clinical info dataset
     :param clinical_info : The clinical info dataset (only the 'time' and 'event' field is needed in this function)
     :param solution_count: The number of solution which is used in ensamble function [Currently we only use 1 solution]
     :param feature_count : The number of features which should be chosen
     :return: New feature dataframe
     '''

    # Call the r wrapper to use mrmre for selecting features
    # This function use mRMR.ensemble to return max relevant min redundant features
    robjects.r('''
                # create a function `mrmrSelection`
                mrmrSelection <- function(df_all, feature_count= feature_count, solution_count = solution_count) {
                    df_all$event <- as.numeric(df_all$event)
                    df_all$surv <- with(df_all, Surv(time, event))
                    df_all <- subset(df_all, select = -c(time,event) )
                    #df_all <- df_all[,colSums(is.na(df_all))<nrow(df_all)]
                    surv_index = grep("surv", colnames(df_all)) 
                    feature_data <- mRMR.data(df_all)
                    feature_selected <- mRMR.ensemble(data = feature_data, target_indices = surv_index,feature_count= feature_count, solution_count = solution_count)
                    result <-  as.data.frame(solutions(feature_selected)[1])

                }
                ''')

    r_mrmrSelection = robjects.r['mrmrSelection']
    # Merge the features and labels (the survival time, which are the 'time' and 'event'  field of clinical information)
    target = target_df.copy()
    if not survival:
        target['time'] = target['target']
        target['event'] = 1

    target.set_index('id', inplace=True)
    features = features.merge(target[['time', 'event']], how='inner', left_index=True, right_index=True)
    all_features = r_mrmrSelection(features, feature_count=feature_count, solution_count=solution_count)
    selected_features = pandas2ri.ri2py_dataframe(all_features)
    selected_features = selected_features.sub(1)
    return selected_features.iloc[:, 0]


def select_mrmr_features(dataframe_features: pd.DataFrame,
                         target_df: pd.DataFrame,
                         mrmr_size: int,
                         survival: bool = False):
    """
      select the mrmr features

      :param dataframe_features: DataFrame of the features
      :param target_df: DataFrame of the target data
      :param mrmr_size: The number of features which should be selected with mrmr
      :param train_ids: List of the train_ids that should be considered in mrmr
      :return: DataFrame that contain selected features
    """
    target = target_df  # clinical_df[train_ids.tolist()]
    features = dataframe_features.T  # todo apply on all the model
    features = variance_threshold_selector(features)
    mrmr_list = mrmr_selection(features=features,
                               clinical_info=target,
                               solution_count=1,
                               feature_count=mrmr_size,
                               survival=survival)

    features = dataframe_features.iloc[mrmr_list]  # todo check iloc is better or loc should check
    # features.to_csv('mrmr.csv')
    return features


def select_valid_features(dataframe_features: pd.DataFrame,
                          target_df: pd.DataFrame):
    features = dataframe_features.T
    features = features.merge(target_df[['time', 'event']], how='inner', left_index=True, right_index=True)
    return variance_threshold_selector(features).drop(['time', 'event'], axis=1).T


if __name__ == "__main__":
    features = pd.read_csv('data/OPC1_radiomic_features.csv')
    clinical_info = pd.read_csv('data/clinical_info.csv')
    solution = select_mrmr_features(features, clinical_info, 10, True)
    print(solution)