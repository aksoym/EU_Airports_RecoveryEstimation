import numpy
import pandas as pd
import numpy as np
from functions import dfFlights_twFilter, flightFlow
import pickle


pickle_path = 'recoveryRate_pickles/Munich_allTime_recoveryRates.pickle'
#apt_name = pickle_path.split('_')[0]
airport_code = 'EDDM'

with open(pickle_path, 'rb') as file:
    recovery_rate_dict = pickle.load(file)

def capacity_change(airport_code, tw, df_flights):

    first_hours_capacity = df_flights[(df_flights['arr'] == airport_code)
                                      & ((df_flights['ftfmArr_tw'] == tw + 0)
                                         | (df_flights['ftfmArr_tw'] == tw + 1)
                                         | (df_flights['ftfmArr_tw'] == tw + 2)
                                         | (df_flights['ftfmArr_tw'] == tw + 3))].__len__()


    third_hours_capacity = df_flights[(df_flights['arr'] == airport_code)
                                      & ((df_flights['ftfmArr_tw'] == tw + 8)
                                         | (df_flights['ftfmArr_tw'] == tw + 9)
                                         | (df_flights['ftfmArr_tw'] == tw + 10)
                                         | (df_flights['ftfmArr_tw'] == tw + 11))].__len__()





    return third_hours_capacity - first_hours_capacity



recovery_rate_capacity_correlation_dict = {}
for day in np.arange('2018-01', '2018-07', dtype='datetime64[D]'):

    day_trimmed_str = np.datetime_as_string(day).replace('-', '')
    df_flights = pd.read_csv('csv/' + day_trimmed_str[0:6] + "/" + day_trimmed_str + ".csv")

    recovery_rate_capacity_correlation_dict[day] = []

    for idx, recovery_rate_value in enumerate(np.nditer(recovery_rate_dict['Unmasked'][day, 0])):
        if not np.isnan(recovery_rate_value):
            recovery_rate_capacity_correlation_dict[day].append((recovery_rate_value.item(0),
                                                                 capacity_change(airport_code, idx*12, df_flights)
                                                                 ))
        else:
            pass

#%%
recovery_rate_capacity_change_list = []

for values in recovery_rate_capacity_correlation_dict.values():
    recovery_rate_capacity_change_list.extend(values)

rr_array = [x[0] for x in recovery_rate_capacity_change_list]
capacity_change_array = [y[1] for y in recovery_rate_capacity_change_list]

rr_capacity_array = np.array([rr_array[1:], capacity_change_array[:-1]])

print(rr_capacity_array)

corrcoef = np.corrcoef(rr_capacity_array)



#%%
import matplotlib.pyplot as plt

print(corrcoef)
