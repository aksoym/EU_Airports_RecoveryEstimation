import pickle
import numpy as np
import pandas as pd
from feature_generating_functions import capacity

with open('rr_feature_df_pickles/rr_feature_df_prev_capacity.pickle', 'rb') as file:
    rr_feature_dataframe = pickle.load(file)

date_tw_indexes = rr_feature_dataframe.index
apt_code = 'EDDM'


avg_capacity_list = []
total_capacity_list = []
for index in date_tw_indexes:
    date = index.split('_')[0]
    trimmed_date = date.replace('-', '')
    tw = int(index.split('_')[1])

    df_flights = pd.read_csv('../csv/' + trimmed_date[0:6] + "/" + trimmed_date + ".csv")

    avg_capacity, total_capacity = capacity(apt_code, tw, df_flights)

    avg_capacity_list.append(avg_capacity)
    total_capacity_list.append(total_capacity)




rr_feature_dataframe['total_capacity'] = total_capacity_list
rr_feature_dataframe['avg_capacity'] = avg_capacity_list

rr_feature_dataframe.to_pickle('rr_feature_df_total_capacity.pickle')