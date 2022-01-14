import numpy
import pandas as pd
import numpy as np
from functions import dfFlights_twFilter, flightFlow
import pickle


file = "../../Arinc/Main/allft_data/csv/201805"
date = 20180529
tw = 2

df_flights = pd.read_csv(file + "/" + str(date) + ".csv")
apt_df_filtered = pd.read_csv("../../Arinc/Main/misc_data/airportFiltered.csv", index_col=0)

pickle_path = 'recoveryRate_pickles/Munich_allTime_recoveryRates.pickle'
#apt_name = pickle_path.split('_')[0]
apt_name = 'Munich'

with open(pickle_path, 'rb') as file:
    recovery_rate_dict = pickle.load(file)

def capacity_change(airport, tw):

    first_hours_capacity = df_flights[(df_flights['arr'] == airport)
                                      & ((df_flights['ftfmArr_tw'] == tw + 0)
                                         | (df_flights['ftfmArr_tw'] == tw + 1)
                                         | (df_flights['ftfmArr_tw'] == tw + 2)
                                         | (df_flights['ftfmArr_tw'] == tw + 3))].__len__()

    second_hours_capacity = df_flights[(df_flights['arr'] == airport)
                                       & ((df_flights['ftfmArr_tw'] == tw + 4)
                                          | (df_flights['ftfmArr_tw'] == tw + 5)
                                          | (df_flights['ftfmArr_tw'] == tw + 6)
                                          | (df_flights['ftfmArr_tw'] == tw + 7))].__len__()

    third_hours_capacity = df_flights[(df_flights['arr'] == airport)
                                      & ((df_flights['ftfmArr_tw'] == tw + 8)
                                         | (df_flights['ftfmArr_tw'] == tw + 9)
                                         | (df_flights['ftfmArr_tw'] == tw + 10)
                                         | (df_flights['ftfmArr_tw'] == tw + 11))].__len__()

    first_change = second_hours_capacity - first_hours_capacity
    second_change = third_hours_capacity - second_hours_capacity
    total_change = first_change + second_change

    return total_change




for day in dates:
    for tw

#%%

recovery_rate_dict['Unmasked'][np.datetime64('2018-01-05'), 0]
