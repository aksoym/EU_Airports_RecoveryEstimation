import torch

from torch.utils.data import DataLoader
from dataset import RecoveryRateDataset
import os
from training_utils import compute_mean_and_std, apply_std_scaling, set_seed
from model_lightning import LSTMEstimator
from training_loop import model_train

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

SEED = 1234
seed_everything(SEED, workers=True )


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
BATCH_SIZE = 32
LR = 0.001
N_EPOCHS = 500

training_data_loader = DataLoader(scaled_training_data, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=2)
val_data_loader = DataLoader(scaled_val_data)
test_data_loader = DataLoader(scaled_test_data)


model = LSTMEstimator(len(rr_dataset.feature_names), initial_dense_layer_size=50, dense_parameter_multiplier=2,
                      dense_layer_count=3, lstm_layer_count=3, lstm_hidden_units=50, sequence_length=SEQUENCE_LENGTH)

wandb_logger = WandbLogger(project='recovery_rate')
checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")

lightning_trainer = Trainer(logger=wandb_logger, callbacks=[checkpoint_callback])
lightning_trainer.fit(model, training_data_loader, val_data_loader)





