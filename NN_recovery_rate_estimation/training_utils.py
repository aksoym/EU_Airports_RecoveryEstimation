import torch
import torch.utils.data
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    random.random(seed)
    np.random.seed(0)


def compute_mean_and_std(dataset_subset):
    feature_tensors = []
    for feature_vector, target in dataset_subset:
        feature_tensors.append(feature_vector)

        if torch.isnan(feature_vector).any() or torch.isnan(target).any():
            raise Exception('NaN value encountered in tensors.')

    all_training_features = torch.cat(feature_tensors, dim=0)
    all_training_features_mean = torch.mean(all_training_features, dim=0)
    all_training_features_std = torch.std(all_training_features, dim=0)

    #scaled_feature_tensors = (all_training_features - all_training_features_mean) / all_training_features_std


    return (all_training_features_mean, all_training_features_std)

def apply_std_scaling(dataset_subset, mean, std):
    feature_tensors = []
    target_tensors = []
    for feature_vector, target in dataset_subset:
        scaled_feature_vector = (feature_vector - mean) / std

        feature_tensors.append(scaled_feature_vector)
        target_tensors.append(target)

    stacked_features = torch.stack(feature_tensors, dim=0)
    stacked_targets = torch.stack(target_tensors, dim=0)

    return torch.utils.data.TensorDataset(stacked_features, stacked_targets)




