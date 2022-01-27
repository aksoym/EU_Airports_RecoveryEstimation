import pandas as pd
import numpy as np


airport_state_df = pd.read_pickle('../data/all_time_airport_state_df.pickle')
print(airport_state_df.describe([0.9, 0.95, 0.99]))