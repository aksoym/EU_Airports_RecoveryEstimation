import pickle
import numpy as np
import pandas as pd
indexer = pd.IndexSlice
from tqdm import tqdm

#Load the whole dict.
with open('../data/all_airports_all_states_dict.pickle', 'rb') as file:
    all_time_dict = pickle.load(file)

#Get airport name list for indexing.
airport_name_list = pd.read_csv('../../misc_data/airportFiltered.csv', index_col=0).index.tolist()

#Get the dates and tws.
date_list = [np.datetime_as_string(date) for date in np.arange(np.datetime64('2018-01-01'),
                                                               np.datetime64('2018-07-01'))]
tw_list = range(24)

#Create the multi index object for dataframe.
dataframe_mult_index = pd.MultiIndex.from_product([date_list, tw_list, airport_name_list],
                                                  names=['date', 'tw', 'apt'])

#Create the respective dataframes.
infection_rate_dataframe = pd.DataFrame(index=dataframe_mult_index, columns=airport_name_list)
flight_flow_dataframe = pd.DataFrame(index=dataframe_mult_index, columns=airport_name_list)

airport_state_dataframe = pd.DataFrame(index=dataframe_mult_index)

rr_vector_list = []
inf_matrix_list = []
ff_matrix_list = []
apt_delay_list = []
#Get each item from the dict and unpack.
for key, value in tqdm(all_time_dict.items()):
    rr_vector, inf_matrix, ff_matrix, delay_df = value
    date = key.split('_')[0]
    tw = key.split('_')[1]

    #For date, tw and all airports, assign recovery rate vector. 133x1
    rr_vector_list.append(rr_vector[:-1])

    #For date, tw and all airports, assign delay values.
    apt_delay_list.append(delay_df.loc[:-1, ['d_0', 'd_0_avg', 'd_0_avg15']])


    #For date, tw assign the the matrices directly. airport_list x airport_list --> 133x133

    inf_matrix_list.append(inf_matrix.iloc[0:-1, 0:-1])
    #Same for flight flow matrix.

    ff_matrix_list.append(ff_matrix.iloc[0:-1, 0:-1])




infection_rate_dataframe.to_pickle('../data/all_time_infection_rate_df.pickle')
flight_flow_dataframe.to_pickle('../data/all_time_flight_flow_df.pickle')
airport_state_dataframe.to_pickle('../data/all_time_airport_state_df.pickle')




