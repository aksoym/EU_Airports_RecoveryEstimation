import numpy as np
import pandas as pd
import plotly.express as px

apt_df_path = "../recovery_rate/recovery_rate_training_dataframes/data/airport_state_weather_prediction_added_rev1.pickle"
infection_matrix_path = "../recovery_rate/recovery_rate_training_dataframes/data/all_time_infection_rate_df_rev1.pickle"

airport_data = pd.read_pickle(apt_df_path)
infection_matrix_data = pd.read_pickle(infection_matrix_path)

#Data preprocess.
#Clip the normalized delay to [0, 1].
airport_data["norm_delay_per_f"] = airport_data["norm_delay_per_f"].clip(lower=0, upper=1.0).values
median_rr_values = airport_data["recovery_rate"].median()
airport_data["recovery_rate"] = airport_data["recovery_rate"].fillna(method="backfill").fillna(median_rr_values).values

#Create a column to store predicted recovery rates.
airport_data["predicted_recovery_rate"] = airport_data["recovery_rate"].values

####TODO load model, load dataset, do inference here.
####
