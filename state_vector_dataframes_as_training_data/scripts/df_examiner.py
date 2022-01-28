import pandas as pd
import numpy as np


airport_state_df = pd.read_pickle('../data/all_airports_all_states_dict.pickle')

print(airport_state_df.values())