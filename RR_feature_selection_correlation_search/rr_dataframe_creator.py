import numpy as np
import pickle

import pandas as pd

pickle_path = '../recoveryRate_pickles/initial_recovery_rate_dicts/Munich_allTime_recoveryRates.pickle'

with open(pickle_path, 'rb') as file:
    recovery_rate_dict = pickle.load(file)


indexes = []
recovery_rates = []
for key, rr_array in recovery_rate_dict['Unmasked'].items():
    for idx, recovery_rate_value in enumerate(rr_array):
        if not np.isnan(recovery_rate_value):
            recovery_rates.append(recovery_rate_value)
            indexes.append(np.datetime_as_string(key[0]) + f"_{idx*12}")



rr_feature_dataframe = pd.DataFrame(
    data = {'recovery_rate': recovery_rates}, index=pd.Index(indexes)
)



with open('rr_feature_df_pickles/rr_zero_feature_dataframe_Munich.pickle', 'wb') as file:
    pickle.dump(rr_feature_dataframe, file)