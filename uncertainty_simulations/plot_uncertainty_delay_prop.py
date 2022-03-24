import numpy as np
import pandas as pd

SEED = 1657
rng = np.random.default_rng(seed=SEED)

actual_arrival_times = rng.integers(15, 150, size=(133, 133))

flow_matrix_path = "../recovery_rate/recovery_rate_training_dataframes/data/all_time_flight_flow_df_rev1.pickle"
flight_flow_data = pd.read_pickle(flow_matrix_path)

random_flow_matrix_pattern = flight_flow_data.groupby("apt").sample()
print(random_flow_matrix_pattern)