import pandas as pd
import numpy as np
import pickle
from feature_generating_functions import demand
from tqdm import tqdm

apt_df_filtered = pd.read_csv('../../misc_data/airportFiltered.csv', index_col=0)

rr_dataframe = pd.read_pickle('../rr_feature_df_pickles/rr_feature_df_w_regulations.pickle')

apt_code = 'EDDM'

date_tw_indexes = rr_dataframe.index

avg_demand_list = []
for index in tqdm(date_tw_indexes):
    date = index.split('_')[0]
    trimmed_date = date.replace('-', '')
    tw = int(index.split('_')[1])

    df_flights = pd.read_csv('../../csv/' + trimmed_date[0:6] + "/" + trimmed_date + ".csv")

    avg_demand, total_demand = demand(apt_code, tw, df_flights, apt_df_filtered)

    avg_demand_list.append(avg_demand)


rr_dataframe['avg_demand'] = avg_demand_list

rr_dataframe.to_pickle('../rr_feature_df_pickles/rr_feature_df_w_demand.pickle')