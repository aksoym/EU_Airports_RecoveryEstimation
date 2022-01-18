import pickle
import numpy as np
import pandas as pd
from feature_generating_functions import capacity_change


with open('../rr_feature_df_pickles/rr_feature_df_prev_total_capacity.pickle', 'rb') as file:
    rr_feature_dataframe = pickle.load(file)

date_tw_indexes = rr_feature_dataframe.index
apt_code = 'EDDM'


capacity_change_list = []
for index in date_tw_indexes:
    date = index.split('_')[0]
    trimmed_date = date.replace('-', '')
    tw = int(index.split('_')[1])

    df_flights = pd.read_csv('../csv/' + trimmed_date[0:6] + "/" + trimmed_date + ".csv")

    capacity_change_list.append(capacity_change(apt_code, tw, df_flights))




rr_feature_dataframe['capacity_change'] = capacity_change_list

rr_feature_dataframe.to_pickle('rr_feature_df_pickles/rr_feature_df_prev_total_capacity.pickle')

