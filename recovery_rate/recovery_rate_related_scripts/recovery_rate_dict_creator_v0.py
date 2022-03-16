import numpy as np
import pandas as pd
from tqdm import tqdm
from functions import recoveryRatePipeline
import pickle

center_apt = 'EDDM'
center_apt_str = 'Munich'

reg_data = pd.read_csv('misc_data/regulationData_wBools.csv', index_col=0)

#Convert date strings to datetime dtype format.
reg_data.loc['date'] = pd.to_datetime(reg_data['date'], format="%Y-%m-%d")

#Apply a time filter to regulation data to include our data's timespan --> 2018/01-06
reg_data = reg_data.loc[(reg_data['date'] >= '2018-01-01') & (reg_data['date'] < '2018-07-01')]

#Filter the regulations to include only the rows that include center airport.
reg_data = reg_data.loc[reg_data['airport/airspaceName'] == center_apt]




#Boolean masks for slicing.
cb_mask = (reg_data['BoolCB'] == True)
fog_mask = (reg_data['BoolFog'] == True)
snow_mask = (reg_data['BoolSnow'] == True)
rain_mask = (reg_data['BoolRain'] == True)
thunder_mask = (reg_data['BoolThunder'] == True)
wind_mask = (reg_data['BoolWind'] == True)
capacity_mask = (reg_data['regulationType'] == 'G - Aerodrome Capacity')

#Create mask and respective name list for easier iteration.
mask_list = [cb_mask, fog_mask, snow_mask, rain_mask, thunder_mask, wind_mask, capacity_mask, 1]
mask_names = ['CB', 'Fog', 'Snow', 'Rain', 'Thunder', 'Wind', 'Capacity', 'Unmasked']

apt_df_filtered = pd.read_csv("misc_data/airportFiltered.csv", index_col=0)

def calculateRecoveryRate(date, start_tw, end_tw, apt_df_filtered, desired_airport):

    date_str_list = str(date).split('-')
    date_str = ''.join(date_str_list)

    file_root = "csv/"
    df_flights = pd.read_csv(file_root + date_str[0:6] + "/" + date_str + ".csv")

    iteration_count = int(np.ceil((end_tw - start_tw) / 12))
    recovery_rates = []
    for iter in range(iteration_count):
        recoveryRate = recoveryRatePipeline(df_flights, apt_df_filtered, start_tw)
        recovery_rates.append(recoveryRate.loc[desired_airport])
        start_tw += 12

    return np.array(recovery_rates)


all_recovery_rates_dict ={}
for mask_name, mask in tqdm(zip(mask_names, mask_list)):
    all_recovery_rates_dict[mask_name] = {}

    if mask_name != 'Unmasked':
        for idx, row in reg_data[mask].iterrows():
            date_id = row['date']
            tw_id = row['startTimeWindow']
            recovery_rate = calculateRecoveryRate(row['date'], row['startTimeWindow'], row['endTimeWindow'],
                                  apt_df_filtered, center_apt)

            all_recovery_rates_dict[mask_name][(date_id, tw_id)] = recovery_rate

    else:
        for date in tqdm(np.arange(np.datetime64('2018-01-01'), np.datetime64('2018-07-01'))):
            recovery_rate = calculateRecoveryRate(date, 0, 96,
                                                  apt_df_filtered, center_apt)

            all_recovery_rates_dict[mask_name][(date, 0)] = recovery_rate

with open(f'recoveryRate_pickles/{center_apt_str}_allTime_recoveryRates.pickle', 'wb') as file:
    pickle.dump(all_recovery_rates_dict, file)