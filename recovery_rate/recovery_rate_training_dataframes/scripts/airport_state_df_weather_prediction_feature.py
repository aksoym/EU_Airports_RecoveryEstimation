import pandas as pd
import numpy as np

airport_df = pd.read_pickle('../data/airport_state_df_w_features.pickle')
airport_df.loc[airport_df['reg_bool_type'].isna(), "reg_bool_type"] = None

#Iterate over unique airport names.
for apt_name in airport_df.reset_index()['apt'].unique().tolist():
    #Multindex slice.
    all_apt_rows = (slice(None), slice(None), [apt_name])
    #Create a column named prediction and assign the NEXT weather condition by shifting the rows by -1.
    airport_df.loc[all_apt_rows, 'weather_prediction'] = airport_df.loc[all_apt_rows, 'reg_bool_type'].shift(-1)
    
#Fill with None in case of mismatches.
airport_df.loc[airport_df['weather_prediction'].isna(), "weather_prediction"] = None

#Drop the None column, because it is inferred fro m when each category equals to zero.
one_hot_encoded_weather_predictions = pd.get_dummies(airport_df.weather_prediction).drop('None', axis=1)
airport_df = pd.concat([airport_df, one_hot_encoded_weather_predictions], axis=1)

airport_df.to_pickle('../data/airport_state_weather_prediction_added.pickle')
