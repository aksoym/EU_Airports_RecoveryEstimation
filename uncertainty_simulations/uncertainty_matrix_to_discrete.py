import numpy as np
from typing import List, Dict
import pandas as pd

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
                    self.distribution_matrix[i, j] = self.rng.normal(mean, std, size=(size,)).clip(min=1)
                    
            #Filling diagonal for 0 because it means self loop.
            np.fill_diagonal(self.distribution_matrix, 0)
            #Clipping sub zero entries as this matrix is for future traffic only.
            #self.distribution_matrix = self.distribution_matrix.clip(min=0)
            sample_list.append(self._empty_arrays_2_zero(self.distribution_matrix))
            
        return sample_list
                
                
    def _initialize_distribution_matrix(self):
        self.distribution_matrix = np.empty((self.diag_length, self.diag_length), dtype=object)
        
    def _empty_arrays_2_zero(self, array):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if isinstance(array[i, j], np.ndarray):
                    if array[i, j].size == 0:
                        array[i, j] = 0
                    else:
                        pass
                else:
                    pass
        return array
            
    @staticmethod
    def _find_max_in_nested_array(array: np.ndarray) -> float:
        max_value = -np.inf
        for element in array.flatten():
            if max_value < np.amax(element):
                max_value = np.amax(element)
        return max_value

def to_flowmatrix(array: np.ndarray) -> List[np.ndarray]:
    global_max = UncertaintyMatrix._find_max_in_nested_array(array)
    interval = 15 #in minutes.
    num_infection_matrix_windows = np.ceil(global_max / interval).astype(int)
    
    infection_matrix_list = [np.zeros(array.shape, dtype=np.int8) for _ in range(num_infection_matrix_windows)]
    
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            
                if isinstance(array[i, j], np.ndarray):
                    for element in array[i, j]:
                        inf_matrix_index = np.floor_divide(element, interval).astype(int)
                        infection_matrix_list[inf_matrix_index][i, j] += 1
                        
                else:
                    if array[i, j] > 0:
                        inf_matrix_index = np.floor_divide(array[i, j], interval).astype(int)
                        infection_matrix_list[inf_matrix_index][i, j] += 1
                
    return infection_matrix_list

def find_max_sublist_length(main_list):
    max_ = -np.inf
    for sublist in main_list:
        if max_ < len(sublist):
            max_ = len(sublist)
    return max_
         
def flowmatrix_sort(flow_matrix_list: List[np.ndarray]) -> Dict:
    flow_matrix_dict = dict()
    for i in range(find_max_sublist_length(flow_matrix_list)):
        flow_matrices = []
        for idx, sublist in enumerate(flow_matrix_list):
            if not flow_matrices:
                try:
                    flow_matrices.append(sublist[i])
                except:
                    pass
            if flow_matrices:
                try:
                    if not np.array_equal(sublist[i], flow_matrices[idx-1]):
                        flow_matrices.append(sublist[i])
                except:
                    pass
        flow_matrix_dict[i] = flow_matrices
    return flow_matrix_dict
     
def flowmat_dict_to_infmat_dict(flow_matrix_dict):
    inf_matrix_dict = dict()
    for key, array_subarray in flow_matrix_dict.items():
        new_subarray_list = []
        for array in array_subarray:
            result = np.divide(array, array.sum(axis=1, keepdims=True), out=np.zeros_like(array), where=(array > 0), casting="unsafe")
            new_subarray_list.append(result)
        inf_matrix_dict[key] = new_subarray_list
    return inf_matrix_dict
    
    
if __name__ == "__main__":
    np.set_printoptions(suppress=True, threshold=2e4)
    flow_matrix_path = "../recovery_rate/recovery_rate_training_dataframes/data/all_time_flight_flow_df_rev1.pickle"
    flight_flow_data = pd.read_pickle(flow_matrix_path)

    uncertainty_matrix = UncertaintyMatrix(flight_flow_data.loc[("2018-02-08", 48, slice(None)), :].values)
    arrays = uncertainty_matrix.draw_sample(100)

    flow_matrix_nested_list = [to_flowmatrix(array) for array in arrays]
    organized_flow_matrix_dict = flowmatrix_sort(flow_matrix_nested_list)
    inf_matrix_dict = flowmat_dict_to_infmat_dict(organized_flow_matrix_dict)
    
        
    
                
