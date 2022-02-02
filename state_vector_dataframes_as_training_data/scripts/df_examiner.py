import os
import networkx as nx
import pandas as pd
import numpy as np



df = pd.read_pickle('../data/all_time_airport_state_df.pickle')



print(df.loc[('2018-01-01', 0)].astype(np.float64).dtypes)

graph = nx.from_pandas_adjacency(df.loc[('2018-01-01', 0)].astype(np.float64), create_using=nx.DiGraph)

print(nx.centrality.eigenvector_centrality(graph).)





def add_centralities(airport_state_df, flight_flow_df):
    eigenvector_centrality_list = []
    degree_centrality_list = []

    for date in np.arange(np.datetime64('2018-01-01'), np.datetime64('2018-07-01')):
        for tw in range(0, 96, 4):
            graph = nx.from_pandas_adjacency(flight_flow_df.loc[(date, tw)].astype(np.float64), create_using=nx.DiGraph)
            eigenvector_centrality_list.append(nx.eigenvector_centrality(graph))
            degree_centrality_list.append(nx.degree_centrality(graph))


    airport_state_df['eigenvector_centrality'] = eigenvector_centrality_list
    airport_state_df['degree_centrality'] = degree_centrality_list

    return airport_state_df