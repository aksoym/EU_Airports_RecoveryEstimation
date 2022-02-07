from collections import OrderedDict
import os

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split


import pytorch_lightning as pl



class LSTMEstimator(pl.LightningModule):

    def __init__(self, feature_size, initial_dense_layer_size, dense_parameter_multiplier, dense_layer_count,
                 lstm_layer_count, lstm_hidden_units, sequence_length, loss='huber'):
        super(LSTMEstimator, self).__init__()

        self.lstm_depth = lstm_layer_count
        self.dense_multiplier = dense_parameter_multiplier
        self.feature_size = feature_size
        self.dense_size = initial_dense_layer_size
        self.dense_count = dense_layer_count
        self.lstm_hidden_count = lstm_hidden_units
        self.window = sequence_length
        self.loss_functions = {'huber': F.huber_loss, 'mse': F.mse_loss}
        self.loss = self.loss_functions[loss]



        layer_list = []
        for layer_count in range(self.dense_count):
            if layer_count == 0:
                input_size = self.feature_size
                self.dense_output_size = self.dense_size

            layers_to_add = [('linear' + str(layer_count), nn.Linear(input_size, self.dense_output_size)),
                             ('relu' + str(layer_count), nn.LeakyReLU())]
            layer_list.extend(layers_to_add)

            input_size = self.dense_output_size
            self.dense_output_size = int(input_size * self.dense_multiplier)

        self.feature_extracting_layers = nn.Sequential(
            OrderedDict(layer_list)
        )

        self.lstm_layer = nn.LSTM(
            int(self.dense_output_size / self.dense_multiplier), self.lstm_hidden_count, dropout=0.5,
            batch_first=True, num_layers=self.lstm_depth
        )

        self.estimator_layers = nn.Sequential(
            nn.Linear(self.lstm_hidden_count * self.window, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        #Apply feature extraction to every vector in the sequence.
        #Input is in the form x = (batch, sequence, features)

        flattened_features = x.reshape(-1, x.shape[-1])
        flattened_extracted_features = self.feature_extracting_layers(flattened_features)
        sequenced_features = flattened_extracted_features.reshape(x.shape[0], self.window, -1)

        lstm_output, _ = self.lstm_layer(sequenced_features)

        #Squeeze sequence and feature dimensions together for estimating layers.
        flattened_lstm_output = lstm_output.reshape(x.shape[0], -1)
        estimator_output = self.estimator_layers(flattened_lstm_output)

        return estimator_output


    def training_step(self, batch, batch_idx):
        feature_sequence, target = batch
        estimation = self(feature_sequence)
        loss = self.loss(estimation, target.reshape(-1, 1))
        self.log('training_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        feature_sequence, target = batch
        estimation = self(feature_sequence)
        loss = self.loss(estimation, target.reshape(-1, 1))
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)