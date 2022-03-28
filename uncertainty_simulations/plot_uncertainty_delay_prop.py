import numpy as np
import pandas as pd
from uncertainty_matrix_to_discrete import UncertaintyMatrix, convert_flowmat_to_infmat, to_flowmatrix
from differential_propagation_model import diff_propagation
import seaborn as sns

SEED = 1657
rng = np.random.default_rng(seed=SEED)

actual_arrival_times = rng.integers(15, 150, size=(133, 133))

flow_matrix_path = "../recovery_rate/recovery_rate_training_dataframes/data/all_time_flight_flow_df_rev1.pickle"
flight_flow_data = pd.read_pickle(flow_matrix_path)

apt_df_path = "../recovery_rate/recovery_rate_training_dataframes/data/airport_state_weather_prediction_added_rev1.pickle"
apt_df = pd.read_pickle(apt_df_path)

#Data preprocess.
#Clip the normalized delay to [0, 1].
apt_df["norm_delay_per_f"] = apt_df["norm_delay_per_f"].clip(lower=0, upper=1.0).values
median_rr_values = apt_df["recovery_rate"].median()
apt_df["recovery_rate"] = apt_df["recovery_rate"].fillna(method="backfill").fillna(median_rr_values).values

random_idx = rng.integers(0, len(flight_flow_data)/133)
random_slice = slice(random_idx*133, (random_idx+1)*133)
random_flow_matrix_pattern = flight_flow_data.iloc[random_slice, :]

uncertainty_mat = UncertaintyMatrix(random_flow_matrix_pattern.values, mean_values=actual_arrival_times)
actual_without_uncertainty = UncertaintyMatrix(random_flow_matrix_pattern.values, mean_values=actual_arrival_times, std_values=np.zeros(shape=(133, 133)))

arrays = uncertainty_mat.draw_sample(100)
flow_matrix_nested_list = [to_flowmatrix(array) for array in arrays]
convert_flowmat_to_infmat(flow_matrix_nested_list)

actual_array = actual_without_uncertainty.draw_sample(1)[0]
actual_flow_matrix = [to_flowmatrix(actual_array)]
convert_flowmat_to_infmat(actual_flow_matrix)


sliced_apt_df = apt_df.iloc[random_slice, :]

p_dot_vector = diff_propagation(sliced_apt_df["norm_delay_per_f"].values, sliced_apt_df["recovery_rate"].values, actual_flow_matrix[0][0])

p_zero = sliced_apt_df["norm_delay_per_f"].values
p_list = [p_zero]
for inf_matrix in actual_flow_matrix[0]:
    print(np.sum(p_list))
    p_dot_vector = diff_propagation(sum(p_list), sliced_apt_df["recovery_rate"].values, inf_matrix)
    p_list.append(p_dot_vector)
    

print(len(p_list), len(np.cumsum(p_list)[0]))
sns.lineplot(x=range(len(p_list)), y=np.cumsum(p_list)[0])





