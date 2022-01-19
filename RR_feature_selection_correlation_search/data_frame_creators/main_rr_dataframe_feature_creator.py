import pandas as pd
import pickle
import numpy as np
from feature_adding_functions import *


recovery_rate_df = pd.read_pickle('../rr_feature_df_pickles/rr_zero_feature_dataframe_Frankfurt.pickle')
apt_code = 'EDDF'

index_list = recovery_rate_df.index.tolist()
date_idx_list = [date_index.split('_')[0] for date_index in index_list]
tw_idx_list = [tw_index.split('_')[1] for tw_index in index_list]

recovery_rate_df['date_idx'] = date_idx_list
recovery_rate_df['tw_idx'] = tw_idx_list


recovery_rate_df = add_regulation_features(recovery_rate_df, apt_code)
recovery_rate_df = add_avg_capacity(recovery_rate_df, apt_code)
recovery_rate_df = add_capacity_change(recovery_rate_df, apt_code)
recovery_rate_df = add_prev_total_capacity(recovery_rate_df, apt_code)
recovery_rate_df = add_prev_capacity_change(recovery_rate_df, apt_code)
recovery_rate_df = add_demand(recovery_rate_df, apt_code)

recovery_rate_df.to_pickle('rr_all_features_df_Frankfurt.pickle')


