import torch
from torch.utils.data import DataLoader
from dataset import RecoveryRateDataset
import os
from scaling_utils import compute_mean_and_std, apply_std_scaling
from model import LSTMEstimator
from training_loop import model_train

root_dir = os.getcwd().split('Start_PyCharm')[0]
data_file_path = root_dir + 'Start_PyCharm/state_vector_dataframes_as_training_data/data/airport_state_df_w_features.pickle'


SEQUENCE_LENGTH = 5

features_to_drop = ['reg_type', 'reg_bool_type', 'reg_cause']
rr_dataset = RecoveryRateDataset(data_file_path, features_to_drop=features_to_drop, sequence_length=SEQUENCE_LENGTH,
                                 fill_with='backfill')


train_val_test_ratios = (0.6, 0.2, 0.2)
train, val, test = torch.utils.data.random_split(rr_dataset, lengths=[int(len(rr_dataset)*ratio)
                                                                      for ratio in train_val_test_ratios])

#Compute mean and std for standard scaling.
training_mean, training_std = compute_mean_and_std(train)

scaled_training_data = apply_std_scaling(train, training_mean, training_std)
scaled_val_data = apply_std_scaling(val, training_mean, training_std)
scaled_test_data = apply_std_scaling(test, training_mean, training_std)


#Training parameters.
BATCH_SIZE = 64
LR = 0.01
N_EPOCHS = 50

training_data_loader = DataLoader(scaled_training_data, batch_size=BATCH_SIZE, shuffle=True)
val_data_loader = DataLoader(scaled_val_data)
test_data_loader = DataLoader(scaled_test_data)

#%%
model = LSTMEstimator(len(rr_dataset.feature_names), initial_dense_layer_size=50, dense_parameter_multiplier=2,
                      dense_layer_count=3, lstm_layer_count=3, lstm_hidden_units=50, sequence_length=SEQUENCE_LENGTH)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
lr_scheduler = None # Optional
criterion = torch.nn.MSELoss()

model_save_path='best_model'

train_history = []
val_history = []

model_train(model, N_EPOCHS, training_data_loader, val_data_loader, train_history, val_history, model_save_path)


