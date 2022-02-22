import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from functions import hourly_flow_matrix

#This is to omit settingwithcopy warnings.
pd.set_option('mode.chained_assignment', None)

#Read airport names.
file_root = "../../csv/"
apt_df_filtered = pd.read_csv("../../misc_data/airportFiltered.csv", index_col=0)

#Create matrix lists and fill them with hourly flow matrices for both departure and arrival.
dep_matrix_list = []
arr_matrix_list = []
#First iterate over dates.
for date in tqdm(np.arange(np.datetime64('2018-01-01'), np.datetime64('2018-07-01'))):
    #Format the datetime string to match with the csv files to read.
    date_str_list = str(date).split('-')
    date_str = ''.join(date_str_list)
    df_flights = pd.read_csv(file_root + date_str[0:6] + "/" + date_str + ".csv")
    
    #Second, iterate over time windows. Timewindows jumps by 3 hours because the function returns 3 hourly matrices.
    for tw in range(0, 96, 12):
        first_dep, second_dep, third_dep, first_arr, second_arr, third_arr = hourly_flow_matrix(df_flights, tw, apt_df_filtered)

        first_dep = first_dep.iloc[:-1, :-1]
        second_dep = second_dep.iloc[:-1, :-1]
        third_dep = third_dep.iloc[:-1, :-1]
        dep_matrix_list.extend([first_dep, second_dep, third_dep])

        first_arr = first_arr.iloc[:-1, :-1]
        second_arr = second_arr.iloc[:-1, :-1]
        third_arr = third_arr.iloc[:-1, :-1]
        arr_matrix_list.extend([first_arr, second_arr, third_arr])


#Date and tw indices for multi index creation.
date_list = [np.datetime_as_string(date) for date in np.arange(np.datetime64('2018-01-01'),
                                                               np.datetime64('2018-07-01'))]
tw_list = range(0, 96, 4)
#Creates the mult index by all possible products of the provided lists..
dataframe_mult_index = pd.MultiIndex.from_product([date_list, tw_list, apt_df_filtered.index.tolist()],
                                                  names=['date', 'tw', 'apt'])

#Empty dataframes.
dep_flight_flow_dataframe = pd.DataFrame(index=dataframe_mult_index, columns=apt_df_filtered.index.tolist())
arr_flight_flow_dataframe = pd.DataFrame(index=dataframe_mult_index, columns=apt_df_filtered.index.tolist())

#Combined the list of dataframes to a single df.
combined_dep_matrix = pd.concat(dep_matrix_list, axis=0, ignore_index=True)
combined_arr_matrix = pd.concat(arr_matrix_list, axis=0, ignore_index=True)

#Assign the values of dataframes to their respective dfs.
dep_flight_flow_dataframe.iloc[0:len(combined_dep_matrix), 0:len(combined_dep_matrix.columns)] = combined_dep_matrix.values
arr_flight_flow_dataframe.iloc[0:len(combined_arr_matrix), 0:len(combined_arr_matrix.columns)] = combined_arr_matrix.values

dep_flight_flow_dataframe.to_pickle('hourly_departure_matrix.pickle')
arr_flight_flow_dataframe.to_pickle('hourly_arrival_matrix.pickle')