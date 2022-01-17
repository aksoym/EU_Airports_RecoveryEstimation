import pickle
import numpy as np
import pandas as pd
from feature_generating_functions import capacity_change



with open('rr_feature_df_pickles/rr_feature_df_prev_total_capacity.pickle', 'rb') as file:
    rr_feature_dataframe = pickle.load(file)

date_tw_indexes = rr_feature_dataframe.index
apt_code = 'EDDM'


prev_capacity_change_list = []
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
        prev_capacity_change_list.append(capacity_change(apt_code, (prev_tw % 96), df_flights))
    else:
        df_flights = pd.read_csv('../csv/' + trimmed_date[0:6] + "/" + trimmed_date + ".csv")
        prev_capacity_change_list.append(capacity_change(apt_code, prev_tw, df_flights))


rr_feature_dataframe['prev_capacity_change'] = [np.nan] + prev_capacity_change_list

rr_feature_dataframe.to_pickle('rr_feature_df_pickles/rr_feature_df_prev_total_capacity.pickle')