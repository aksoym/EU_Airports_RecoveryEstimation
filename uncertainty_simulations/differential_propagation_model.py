from copy import deepcopy
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm


def diff_propagation(p_state_vector: np.ndarray, recovery_rate_vector: np.ndarray, infection_rate_matrix: np.ndarray) -> np.ndarray:
    recovery_rate_diag = np.diag(recovery_rate_vector)
    #Diff. SIS. Equation from Nowzar, Preciado, 2016 - Analysis and Control of Epidemics
    #The equation is --> p_dot = (B - D)*p + h
    #Where B is infection matrix, D is recovery rate vector as a diagonal matrix,
    #p is the initial state vector and h is == Bxp.*p (first b and p matrix mult. then p with elementwise mult.)
    p_dot_vector = ((infection_rate_matrix - recovery_rate_diag) @ p_state_vector) - ((infection_rate_matrix @ p_state_vector) * p_state_vector)
    
    return p_dot_vector

def calculate_pdots_on_df(airport_df: pd.DataFrame, infection_matrix: pd.DataFrame, recovery_rate_label, output_label) -> pd.DataFrame:
    df = deepcopy(airport_df)
    df["norm_delay_per_f"] = df["norm_delay_per_f"].clip(lower=0, upper=1.0).values
    for apt in df.index.unique("apt"):
        index = (slice(None), slice(None), [apt])
        df.loc[index, recovery_rate_label] = df.loc[index, recovery_rate_label].fillna(method="backfill")
        df.loc[index, recovery_rate_label] = df.loc[index, recovery_rate_label].fillna(df.loc[index, recovery_rate_label].median())
    
    p_dot_ndarray_list = []
    for date in tqdm(df.index.unique("date")):
        for tw in df.index.unique("tw"):
            index = (date, tw, slice(None))
            recovery_rate_vector = df.loc[index, recovery_rate_label].values
            infection_matrix_ = infection_matrix.loc[index, :].values
            p_state_vector = df.loc[index, "norm_delay_per_f"].values
            p_dot_vector = diff_propagation(p_state_vector, recovery_rate_vector, infection_matrix_)
            p_dot_ndarray_list.append(p_dot_vector)
            
    df[output_label] = np.concatenate(p_dot_ndarray_list, axis=0)
    
    return df
    
if __name__ == "main":
    
    apt_df_path = "../recovery_rate/recovery_rate_training_dataframes/data/airport_state_weather_prediction_added_rev1.pickle"
    flight_flow_path = "../recovery_rate/recovery_rate_training_dataframes/data/all_time_flight_flow_df_rev1.pickle"
    infection_matrix_path = "../recovery_rate/recovery_rate_training_dataframes/data/all_time_infection_rate_df_rev1.pickle"

    airport_df = pd.read_pickle(apt_df_path)
    infection_matrix_df = pd.read_pickle(infection_matrix_path)

    date = "2018-02-12"
    sample_index = (date, slice(16, 88), ["EGLL"])

    # p_vector = np.clip(airport_df.loc[(date, 16, slice(None)), "norm_delay_per_f"].values, 0, 1)
    # p_vector_history = [p_vector]
    # for tw in range(16, 88, 4):
    #     index = (date, tw, slice(None))
        
    #     recovery_rate_vector = airport_df.loc[index, "recovery_rate"].fillna(0).values
    #     infection_matrix = infection_matrix_df.loc[index, :].values
        
    #     p_dot_vector = diff_propagation(p_vector, recovery_rate_vector, infection_matrix)
    #     #Divide the update vector by 3 because we use 3hr long parameters, yet we update hourly.
    #     p_vector_temp = np.column_stack((p_vector, (p_dot_vector / 3)))
    #     p_vector = np.clip(np.sum(p_vector_temp, axis=1), 0, 1)
    #     p_vector_history.append(p_vector)


    # sliced_df = deepcopy(airport_df.loc[sample_index, :])

    # p_vectors = np.column_stack(tuple(p_vector_history))[0, :].astype(np.float64)
    # sliced_df["calculated_delay"] = p_vectors #savgol_filter(p_vectors, 7, 3)

    # fig = px.line(sliced_df.reset_index(), y=["norm_delay_per_f", "calculated_delay"], text="recovery_rate")
    # #fig.add_trace(go.Scatter(x=list(range(0, 24, 3)), y=p_vector.tolist(), mode="lines"))
    # fig.show()
                
    new_df = calculate_pdots_on_df(airport_df, infection_matrix_df)
    print(new_df["p_dot"])
        
        
        





