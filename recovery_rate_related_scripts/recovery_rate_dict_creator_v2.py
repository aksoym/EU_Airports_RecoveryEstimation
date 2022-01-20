### IMPROVED RECOVERY RATE DICTIONARY CREATOR
### OPTIMIZED FOR SPEED FOR LARGE ARRAY AND SEQUENTIAL OPERATIONS
### dictionary key format changed from (np.datetime, tw) to (date:str, tw)
import numpy as np
import pandas as pd
from tqdm import tqdm

from functions import recoveryRatePipeline


apt_df_filtered = pd.read_csv("../misc_data/airportFiltered.csv", index_col=0)

all_parameters_all_airports_dict = {}

for date in tqdm(np.arange(np.datetime64('2018-01-01'), np.datetime64('2018-07-01'))):
    date_str_list = str(date).split('-')
    date_str = ''.join(date_str_list)

    file_root = "../csv/"
    df_flights = pd.read_csv(file_root + date_str[0:6] + "/" + date_str + ".csv")


    for tw in range(0, 96, 4):
        recoveryRates, infectionRates, flight_flow, apt_delay_values = recoveryRatePipeline(df_flights, apt_df_filtered,
                                                                                            tw)
        dict_key = np.datetime_as_string(date) + '_' + str(tw)
        all_parameters_all_airports_dict[dict_key] = (recoveryRates, infectionRates, flight_flow, apt_delay_values)