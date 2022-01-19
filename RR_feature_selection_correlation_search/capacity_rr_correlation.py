import numpy
import pandas as pd
import numpy as np
from functions import dfFlights_twFilter, flightFlow
import pickle


pickle_path = '/RR_feature_selection_correlation_search/rr_feature_df_pickles/rr_dataframe_munich/rr_feature_df_prev_total_capacity.pickle'
#apt_name = pickle_path.split('_')[0]
airport_code = 'EDDM'

with open(pickle_path, 'rb') as file:
    recovery_rate_df = pickle.load(file)

import seaborn as sns
import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
sns.heatmap(recovery_rate_df.loc[:, ['recovery_rate',
                                     'capacity_change',
                                     'prev_capacity_change',
                                     'avg_capacity',
                                     'prev_avg_capacity']].corr(), annot=True, square=True)
plt.subplots_adjust(left=0.2, bottom=0.1)
plt.xticks(rotation=25)
plt.yticks(rotation=25)
plt.savefig('feature_corr_matrix_heatmap.png', dpi=300)
plt.show()