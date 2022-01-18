import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta

regulation_data = pd.read_csv('/home/aksoy/Desktop/START/Muhammet/Start_PyCharm/misc_data/regulationData_wBools.csv')
regulation_data = regulation_data[(regulation_data['date'] >= '2018-01-01') & (regulation_data['date'] <= '2018-07-01')]


print(regulation_data['startTimeWindow'].value_counts(), regulation_data['endTimeWindow'].value_counts())

rr_dataframe = pd.read_pickle('/home/aksoy/Desktop/START/Muhammet/Start_PyCharm/RR_feature_selection_correlation_search/rr_feature_df_pickles/rr_feature_df_date_tw.pickle')

print(rr_dataframe.columns)


regulation_list = []

for date, tw in zip(rr_dataframe['date_idx'], rr_dataframe['tw_idx']):
    date_filtered = regulation_data[regulation_data['date'] == date]
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

rr_dataframe['reg_type'] = reg_type_list
rr_dataframe['reg_cause'] = reg_cause_list
rr_dataframe['reg_bool_type'] = reg_bool_list


#rr_dataframe.to_pickle('../rr_feature_df_pickles/rr_feature_df_w_regulations.pickle')
