from copy import deepcopy
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm


def diff_propagation(p_state_vector: np.ndarray, recovery_rate_vector: np.ndarray, infection_rate_matrix: np.ndarray, timestep=1) -> np.ndarray:
    
    if len(p_state_vector) != 133:
        p_state_vector_list = [p_state_vector[0]]
        for recovery_rate, inf_rate_matrix in zip(recovery_rate_vector, infection_rate_matrix):

            p_state_vector = p_state_vector_list[-1]
            #print(recovery_rate.shape, inf_rate_matrix.shape, p_state_vector.shape)
            recovery_rate_diag = np.diag(recovery_rate)
            p_dot_vector = ((inf_rate_matrix - recovery_rate_diag) @ p_state_vector) - ((inf_rate_matrix @ p_state_vector) * p_state_vector)
            
            p_vector_temp = np.column_stack((p_state_vector, (p_dot_vector / timestep)))
            p_state_vector = np.clip(np.sum(p_vector_temp, axis=1), 0, 1)
            
            p_state_vector_list.append(p_state_vector)
            
        return p_state_vector_list
    
    elif p_state_vector.size == 133:
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

    
if __name__ == "__main__":
    
    from uncertainty_matrix_to_discrete import UncertaintyMatrix, convert_flowmat_to_infmat
    rng = np.random.default_rng()
    apt_df_path = "../recovery_rate/recovery_rate_training_dataframes/data/airport_state_weather_prediction_added_rev1.pickle"
    flight_flow_path = "../recovery_rate/recovery_rate_training_dataframes/data/all_time_flight_flow_df_rev1.pickle"
    infection_matrix_path = "../recovery_rate/recovery_rate_training_dataframes/data/all_time_infection_rate_df_rev1.pickle"

    airport_df = pd.read_pickle(apt_df_path)
    infection_matrix_df = pd.read_pickle(infection_matrix_path)
    flow_matrix_df = pd.read_pickle(flight_flow_path)

    date = "2018-02-27"
    sample_index = (date, slice(24, 92), ["EDDM"])
    
    p_vector = np.clip(airport_df.loc[(date, 24, slice(None)), "norm_delay_per_f"].values, 0, 1)
    p_vector_history = [p_vector]
    for tw in range(24, 92, 4):
        index = (date, tw, slice(None))
        
        recovery_rate_vector = airport_df.loc[index, "recovery_rate"].fillna(0).values
        infection_matrix = infection_matrix_df.loc[index, :].values
        
        p_dot_vector = diff_propagation(p_vector, recovery_rate_vector, infection_matrix)
        #Divide the update vector by 3 because we use 3hr long parameters, yet we update hourly.
        p_vector_temp = np.column_stack((p_vector, (p_dot_vector / 2)))
        p_vector = np.clip(np.sum(p_vector_temp, axis=1), 0, 1)
        p_vector_history.append(p_vector)

    p_vector = np.clip(airport_df.loc[(date, 24, slice(None)), "norm_delay_per_f"].values, 0, 1)
    p_vector_results = []
    for tw in range(24, 92, 4):
        index = (date, tw, slice(None))
        
        recovery_rate_vector = airport_df.loc[index, "recovery_rate"].fillna(0).values
        infection_matrix_layout = flow_matrix_df.loc[index, :].values
        uncertain_matrix = UncertaintyMatrix(infection_matrix_layout, mean_values=rng.integers(10, 50, size=(133, 133)), interval=15)
        arrays = uncertain_matrix.draw_sample(1)
        flow_matrix_nested_list = [UncertaintyMatrix._to_flowmatrix(array, interval=15) for array in arrays]
        convert_flowmat_to_infmat(flow_matrix_nested_list)
        inf_matrices = flow_matrix_nested_list[0][0:4]
        
        result = diff_propagation([p_vector], [recovery_rate_vector]*4, inf_matrices, timestep=6)

        p_vector_results.extend(result)
        p_vector = p_vector_results[-1]

    sample_result = [p_state_vector[6] for p_state_vector in p_vector_results[:-13]]
    
    #print(p_vector_results)
    sliced_df = deepcopy(airport_df.loc[sample_index, :])
    p_vectors = np.column_stack(tuple(p_vector_history))[6, :].astype(np.float64)
    
    sliced_df["calculated_delay"] = p_vectors #savgol_filter(p_vectors, 7, 3)

    fig = px.line(sliced_df.reset_index(), y=["norm_delay_per_f", "calculated_delay"], text="recovery_rate")
    fig.update_layout(yaxis_range=[0,1])
    fig.add_trace(go.Scatter(x=list(np.arange(0, 18, 0.25)), y=sample_result, mode="lines"))
    fig.show()
                
    
        
        
        





