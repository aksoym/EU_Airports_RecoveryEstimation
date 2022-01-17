import pickle
import numpy as np
import pandas as pd
from feature_generating_functions import capacity
import matplotlib.pyplot as plt

with open('rr_feature_df_pickles/rr_feature_df_total_capacity.pickle', 'rb') as file:
    rr_feature_dataframe = pickle.load(file)

date_tw_indexes = rr_feature_dataframe.index
apt_code = 'EDDM'

prev_avg_capacity_list = []
prev_total_capacity_list = []
for idx_of_index, index in enumerate(date_tw_indexes[1:]):
    date = index.split('_')[0]
    trimmed_date = date.replace('-', '')
    tw = int(index.split('_')[1])
    prev_tw = tw - 12

    #Check if previous time window is on a previous date.
    if prev_tw < 0:
        prev_date = date_tw_indexes[idx_of_index - 1]
        prev_date = index.split('_')[0]
        prev_trimmed_date = date.replace('-', '')

        df_flights = pd.read_csv('../csv/' + prev_trimmed_date[0:6] + "/" + prev_trimmed_date + ".csv")
        avg_capacity, total_capacity = capacity(apt_code, (prev_tw % 96), df_flights)

        prev_avg_capacity_list.append(avg_capacity)
        prev_total_capacity_list.append(total_capacity)
    else:
        df_flights = pd.read_csv('../csv/' + trimmed_date[0:6] + "/" + trimmed_date + ".csv")
        avg_capacity, total_capacity = capacity(apt_code, (prev_tw), df_flights)

        prev_avg_capacity_list.append(avg_capacity)
        prev_total_capacity_list.append(total_capacity)


rr_feature_dataframe['prev_total_capacity'] = [np.nan] + prev_total_capacity_list
rr_feature_dataframe['prev_avg_capacity'] = [np.nan] + prev_avg_capacity_list

rr_feature_dataframe.to_pickle('rr_feature_df_pickles/rr_feature_df_prev_total_capacity.pickle')

