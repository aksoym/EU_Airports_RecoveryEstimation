import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm

from functions import flow_matrix_from_departures_only


#This is to omit settingwithcopy warnings.
pd.set_option('mode.chained_assignment', None)


file_root = "../../csv/"
apt_df_filtered = pd.read_csv("../../misc_data/airportFiltered.csv", index_col=0)

matrix_list = []
for date in tqdm(np.arange(np.datetime64('2018-01-01'), np.datetime64('2018-07-01'))):
    date_str_list = str(date).split('-')
    date_str = ''.join(date_str_list)
    df_flights = pd.read_csv(file_root + date_str[0:6] + "/" + date_str + ".csv")

    for tw in range(0, 96, 12):
        first, second, third = flow_matrix_from_departures_only(df_flights, tw, apt_df_filtered)
        first = first.iloc[:-1, :-1]
        second = second.iloc[:-1, :-1]
        third = third.iloc[:-1, :-1]
        matrix_list.extend([first, second, third])


with open('hourly_departure_matrix_list.pickle', 'xb') as file:
    pickle.dump(matrix_list, file)


date_list = [np.datetime_as_string(date) for date in np.arange(np.datetime64('2018-01-01'),
                                                               np.datetime64('2018-07-01'))]
tw_list = range(0, 96, 4)
#Create the multi index object for dataframe.
dataframe_mult_index = pd.MultiIndex.from_product([date_list, tw_list, apt_df_filtered.index.tolist()],
                                                  names=['date', 'tw', 'apt'])
flight_flow_dataframe = pd.DataFrame(index=dataframe_mult_index, columns=apt_df_filtered.index.tolist())


combined_matrix_df = pd.concat(matrix_list, axis=0, ignore_index=True)

flight_flow_dataframe.iloc[0:len(combined_matrix_df), 0:len(combined_matrix_df.columns)] = combined_matrix_df.values

flight_flow_dataframe.to_pickle('hourly_departure_matrix.pickle')