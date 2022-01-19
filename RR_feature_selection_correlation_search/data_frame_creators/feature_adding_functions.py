import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from tqdm import tqdm
from feature_generating_functions import *

apt_df_filtered = pd.read_csv('/home/aksoy/Desktop/START/Muhammet/Start_PyCharm/misc_data/airportFiltered.csv', index_col=0)

#Regulation data for reg. data adding function.
regulation_data = pd.read_csv('/home/aksoy/Desktop/START/Muhammet/Start_PyCharm/misc_data/regulationData_wBools.csv')
regulation_data = regulation_data[(regulation_data['date'] >= '2018-01-01') & (regulation_data['date'] <= '2018-07-01')]


def add_regulation_features(recovery_rate_df, apt_code):
    regulation_list = []

    for date, tw in zip(recovery_rate_df['date_idx'], recovery_rate_df['tw_idx']):
        date_filtered = regulation_data[(regulation_data['date'] == date) & (regulation_data['airport/airspaceName'] == apt_code)]
        tw = float(tw)
        datetime_date = datetime.strptime(date, "%Y-%m-%d")
        next_day = datetime_date + timedelta(days=1)
        next_day_str = datetime.strftime(next_day, "%Y-%m-%d")


        if len(date_filtered):
            tuple_to_append = ('None', 'None', 'None')
            for idx, row in date_filtered.iterrows():

                #A very bad implementation.
                if row['BoolCB'] is True:
                    reg_bool = 'CB'
                elif row['BoolFog'] is True:
                    reg_bool = 'Fog'
                elif row['BoolSnow'] is True:
                    reg_bool = 'Snow'
                elif row['BoolRain'] is True:
                    reg_bool = 'Rain'
                elif row['BoolThunder'] is True:
                    reg_bool = 'Thunder'
                elif row['BoolWind'] is True:
                    reg_bool = 'Wind'
                else:
                    reg_bool = 'None'

                if (row['startTimeWindow'] <= tw) and (tw <= row['endTimeWindow']):

                    tuple_to_append = (row['regulationType'], row['regulationCause'], reg_bool)



                #elif (tw <= row['endTimeWindow'])


            regulation_list.append(tuple_to_append)

        else:

            regulation_list.append(('None', 'None', 'None'))




    reg_type_list = [tup[0] for tup in regulation_list]
    reg_cause_list = [tup[1] for tup in regulation_list]
    reg_bool_list = [tup[2] for tup in regulation_list]

    recovery_rate_df['reg_type'] = reg_type_list
    recovery_rate_df['reg_cause'] = reg_cause_list
    recovery_rate_df['reg_bool_type'] = reg_bool_list

    return recovery_rate_df


def add_capacity_change(recovery_rate_df, apt_code):
    date_tw_indexes = recovery_rate_df.index
    capacity_change_list = []
    for index in date_tw_indexes:
        date = index.split('_')[0]
        trimmed_date = date.replace('-', '')
        tw = int(index.split('_')[1])

        df_flights = pd.read_csv('../../csv/' + trimmed_date[0:6] + "/" + trimmed_date + ".csv")

        capacity_change_list.append(capacity_change(apt_code, tw, df_flights))

    recovery_rate_df['capacity_change'] = capacity_change_list

    return recovery_rate_df


def add_prev_total_capacity(recovery_rate_df, apt_code):
    date_tw_indexes = recovery_rate_df.index

    prev_avg_capacity_list = []
    prev_total_capacity_list = []
    for idx_of_index, index in enumerate(date_tw_indexes[1:]):
        date = index.split('_')[0]
        trimmed_date = date.replace('-', '')
        tw = int(index.split('_')[1])
        prev_tw = tw - 12

        # Check if previous time window is on a previous date.
        if prev_tw < 0:
            prev_date = date_tw_indexes[idx_of_index - 1]
            prev_date = index.split('_')[0]
            prev_trimmed_date = date.replace('-', '')

            df_flights = pd.read_csv('../../csv/' + prev_trimmed_date[0:6] + "/" + prev_trimmed_date + ".csv")
            avg_capacity, total_capacity = capacity(apt_code, (prev_tw % 96), df_flights)

            prev_avg_capacity_list.append(avg_capacity)
            prev_total_capacity_list.append(total_capacity)
        else:
            df_flights = pd.read_csv('../../csv/' + trimmed_date[0:6] + "/" + trimmed_date + ".csv")
            avg_capacity, total_capacity = capacity(apt_code, (prev_tw), df_flights)

            prev_avg_capacity_list.append(avg_capacity)
            prev_total_capacity_list.append(total_capacity)

    recovery_rate_df['prev_total_capacity'] = [np.nan] + prev_total_capacity_list
    recovery_rate_df['prev_avg_capacity'] = [np.nan] + prev_avg_capacity_list

    return recovery_rate_df


def add_prev_capacity_change(recovery_rate_df, apt_code):
    date_tw_indexes = recovery_rate_df.index

    prev_capacity_change_list = []
    for idx_of_index, index in enumerate(date_tw_indexes[1:]):
        date = index.split('_')[0]
        trimmed_date = date.replace('-', '')
        tw = int(index.split('_')[1])
        prev_tw = tw - 12

        # Check if previous time window is on a previous date.
        if prev_tw < 0:
            prev_date = date_tw_indexes[idx_of_index - 1]
            prev_date = index.split('_')[0]
            prev_trimmed_date = date.replace('-', '')

            df_flights = pd.read_csv('../../csv/' + prev_trimmed_date[0:6] + "/" + prev_trimmed_date + ".csv")
            prev_capacity_change_list.append(capacity_change(apt_code, (prev_tw % 96), df_flights))
        else:
            df_flights = pd.read_csv('../../csv/' + trimmed_date[0:6] + "/" + trimmed_date + ".csv")
            prev_capacity_change_list.append(capacity_change(apt_code, prev_tw, df_flights))

    recovery_rate_df['prev_capacity_change'] = [np.nan] + prev_capacity_change_list

    return recovery_rate_df

def add_avg_capacity(recovery_rate_df, apt_code):
    date_tw_indexes = recovery_rate_df.index

    avg_capacity_list = []
    total_capacity_list = []
    for index in date_tw_indexes:
        date = index.split('_')[0]
        trimmed_date = date.replace('-', '')
        tw = int(index.split('_')[1])

        df_flights = pd.read_csv('../../csv/' + trimmed_date[0:6] + "/" + trimmed_date + ".csv")

        avg_capacity, total_capacity = capacity(apt_code, tw, df_flights)

        avg_capacity_list.append(avg_capacity)
        total_capacity_list.append(total_capacity)




    recovery_rate_df['total_capacity'] = total_capacity_list
    recovery_rate_df['avg_capacity'] = avg_capacity_list

    return recovery_rate_df


def add_demand(recovery_rate_df, apt_code):
    date_tw_indexes = recovery_rate_df.index

    avg_demand_list = []
    for index in tqdm(date_tw_indexes):
        date = index.split('_')[0]
        trimmed_date = date.replace('-', '')
        tw = int(index.split('_')[1])

        df_flights = pd.read_csv('../../csv/' + trimmed_date[0:6] + "/" + trimmed_date + ".csv")

        avg_demand, total_demand = demand(apt_code, tw, df_flights, apt_df_filtered)

        avg_demand_list.append(avg_demand)

    recovery_rate_df['avg_demand'] = avg_demand_list

    return recovery_rate_df

