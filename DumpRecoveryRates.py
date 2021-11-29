import pandas as pd
import numpy as np
import pickle
import copy
from functions import recoveryRatePipeline

verbose = 0
#Airspace/airport name to focus. We use ICAO code for data, string form for __repr__, plot purposes.
center_apt = 'EHAM'
center_apt_str = 'Schiphol'


reg_data = pd.read_csv('../../Arinc/Main/misc_data/regulationData_wBools.csv', index_col=0)

#Convert date strings to datetime dtype format.
reg_data.loc['date'] = pd.to_datetime(reg_data['date'], format="%Y-%m-%d")

#Apply a time filter to regulation data to include our data's timespan --> 2018/01-06
reg_data = reg_data.loc[(reg_data['date'] >= '2018-01-01') & (reg_data['date'] < '2018-07-01')]

#Get airportname, reg_count pairs for airports with top regulation counts.
airports_with_top_reg_counts = reg_data.loc[reg_data['airport/airspace'] == 'Aerodrome',
             ['airport/airspaceName']].value_counts().nlargest(10)


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
mask_list = [cb_mask, fog_mask, snow_mask, rain_mask, thunder_mask, wind_mask, capacity_mask]
mask_names = ['CB', 'Fog', 'Snow', 'Rain', 'Thunder', 'Wind', 'Capacity']


#Output some statistics if verbose.
if verbose:
    print("-"*30,
          f"Number of regulation instances: {len(reg_data)}",
          f"\nAirports with the highest regulation counts: \n{airports_with_top_reg_counts}",
          sep='\n')

    #See what type of regulations has occured at what frequencies.
    print("-"*30, 'Regulation type counts:', reg_data['regulationType'].value_counts(), sep='\n')

    masked_weather_reg_count = 0
    for mask, mask_name in zip(mask_list, mask_names):
        print(f'Number of {mask_name} regulation type: {len(reg_data[mask])}')
        masked_weather_reg_count += len(reg_data[mask])

    unmasked_weather_reg_count = len(reg_data.loc[reg_data['regulationType'] == 'W - Weather']) \
                                 - masked_weather_reg_count

    print("-"*30, f"\n Number of weather regulations with no label: {unmasked_weather_reg_count}", sep='\n')

    #See if there is a case where a boolean weather mask is True, but regulation is not weather.
    print("-"*30, reg_data.loc[cb_mask | fog_mask | snow_mask | rain_mask | thunder_mask | wind_mask]['regulationType'].value_counts(), sep='\n')


#We'll be dealing with flight data to calculate the recovery rates, regulation data is only needed
#for us to know which day and which time window to look at. So we get day and tw pairs and then calculate
#the whole network's properties for each pair.


apt_df_filtered = pd.read_csv("../../Arinc/Main/misc_data/airportFiltered.csv", index_col=0)

def calculateRecoveryRate(date, start_tw, end_tw, apt_df_filtered, desired_airport):

    date_str_list = str(date).split('-')
    date_str = ''.join(date_str_list)

    file_root = "../../Arinc/Main/allft_data/csv/"
    df_flights = pd.read_csv(file_root + date_str[0:6] + "/" + date_str + ".csv")

    iteration_count = int(np.ceil((end_tw - start_tw) / 12))
    recovery_rates = []
    for iter in range(iteration_count):
        recoveryRate = recoveryRatePipeline(df_flights, apt_df_filtered, start_tw)
        recovery_rates.append(recoveryRate.loc[desired_airport])
        start_tw += 12

    return np.array(recovery_rates)

all_recovery_rates_dict = {}
for mask_name, mask in zip(mask_names, mask_list):
    all_recovery_rates_dict[mask_name] = {}

    for idx, row in reg_data[mask].iterrows():
        date_id = row['date']
        tw_id = row['startTimeWindow']
        recovery_rate = calculateRecoveryRate(row['date'], row['startTimeWindow'], row['endTimeWindow'],
                              apt_df_filtered, center_apt)

        all_recovery_rates_dict[mask_name][(date_id, tw_id)] = recovery_rate

with open(f'{center_apt_str}_recovery_rates_dict_w_{len(mask_names)}masks.pickle', 'wb') as file:
    pickle.dump(all_recovery_rates_dict, file)

