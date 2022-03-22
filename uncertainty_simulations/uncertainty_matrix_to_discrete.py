import numpy as np
from typing import List
import pandas as pd

flow_matrix_path = "../recovery_rate/recovery_rate_training_dataframes/data/all_time_flight_flow_df_rev1.pickle"
flight_flow_data = pd.read_pickle(flow_matrix_path)

class UncertaintyMatrix():
    
    def __init__(self, flight_flow_matrix: np.ndarray,
                 diag_length: int = 133, 
                 mean_values: np.ndarray = None, 
                 std_values: np.ndarray = None,
                 seed: int = None) -> np.ndarray:
        
        self.flow_matrix = flight_flow_matrix
        self.rng = np.random.default_rng(seed=seed)
        self.diag_length = diag_length
        
        if mean_values is None:
            self.mean_values = self.rng.integers(5, 180, size=(133, 133))
        if std_values is None:
            self.std_values = self.rng.uniform(0.1, 20, size=(133, 133))
            
        
    def draw_sample(self, times: int=1) -> List[np.ndarray]:
        
        sample_list = []
        for _ in range(times):
            self._initialize_distribution_matrix()
            
            for i in range(self.distribution_matrix.shape[0]):
                for j in range(self.distribution_matrix.shape[1]):
                    mean = self.mean_values[i, j]
                    std = self.std_values[i, j]
                    size = self.flow_matrix[i, j]
                    self.distribution_matrix[i, j] = self.rng.normal(mean, std, size=(size,)).clip(min=0)
                    
            #Filling diagonal for 0 because it means self loop.
            np.fill_diagonal(self.distribution_matrix, 0)
            #Clipping sub zero entries as this matrix is for future traffic only.
            #self.distribution_matrix = self.distribution_matrix.clip(min=0)
            sample_list.append(self.distribution_matrix)
            
        return sample_list
                
                
    def _initialize_distribution_matrix(self):
        self.distribution_matrix = np.empty((self.diag_length, self.diag_length), dtype=object)
            
        
uncertainty_matrix = UncertaintyMatrix(flight_flow_data.loc[("2018-02-08", 48, slice(None)), :].values)
print(uncertainty_matrix.draw_sample(1))
