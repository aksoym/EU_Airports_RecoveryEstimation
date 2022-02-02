import pandas
import pandas as pd
from RR_feature_selection_correlation_search.data_frame_creators.feature_adding_functions import *

pandas.set_option('display.max_columns', 15)
from tqdm import tqdm


airport_state_df = pd.read_pickle('../data/all_time_airport_state_df.pickle')
flight_flow_df = pd.read_pickle('../data/all_time_flight_flow_df.pickle')

def add_centralities(airport_state_df, flight_flow_df):
    eigenvector_centrality_list = []
    degree_centrality_list = []

    for date in tqdm(np.arange(np.datetime64('2018-01-01'), np.datetime64('2018-07-01')), position = 0, leave = True):
        for tw in range(0, 96, 4):
            graph = nx.from_pandas_adjacency(flight_flow_df.loc[(np.datetime_as_string(date), tw)].astype(np.float64), create_using=nx.DiGraph)
            eigenvector_centrality_list.extend(list(nx.eigenvector_centrality(graph, max_iter=2000).values()))
            degree_centrality_list.extend(list(nx.degree_centrality(graph).values()))


    airport_state_df['eigenvector_centrality'] = eigenvector_centrality_list
    airport_state_df['degree_centrality'] = degree_centrality_list

    return airport_state_df


feature_functions = (add_centralities, add_outflow, add_demand, add_capacity, add_regulation_features)


for idx, function in tqdm(enumerate(feature_functions)):
    if idx == 0:
        airport_state_df = function(airport_state_df, flight_flow_df)
    else:
        airport_state_df = function(airport_state_df)


print(airport_state_df.head())

airport_state_df.to_pickle('airport_state_df_w_features.pickle')



