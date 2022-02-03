import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict

torch.manual_seed(1)


class LSTMEstimator(nn.Module):

    def __init__(self, feature_size, initial_dense_layer_size, dense_parameter_multiplier, dense_layer_count,
                 lstm_layer_count, lstm_hidden_units, sequence_length):
        super(LSTMEstimator, self).__init__()

        self.lstm_depth = lstm_layer_count
        self.dense_multiplier = dense_parameter_multiplier
        self.feature_size = feature_size
        self.dense_size = initial_dense_layer_size
        self.dense_count = dense_layer_count
        self.lstm_hidden_count = lstm_hidden_units
        self.window = sequence_length


        layer_list = []
        for layer_count in range(self.dense_count):
            if layer_count == 0:
                input_size = self.feature_size
                self.dense_output_size = self.dense_size

            layers_to_add = [('linear'+str(layer_count), nn.Linear(input_size, self.dense_output_size)),
                             ('relu' + str(layer_count), nn.LeakyReLU())]
            layer_list.extend(layers_to_add)

            input_size = self.output_size
            self.dense_output_size = int(input_size * self.dense_multiplier)

        self.feature_extracting_layers = nn.Sequential(
            OrderedDict(layer_list)
        )

        self.lstm_layer = nn.LSTM(
            self.dense_output_size, self.lstm_hidden_count, dropout=0.5,
            batch_first=True, num_layers=self.lstm_depth
        )

        self.estimator_layers = nn.Sequential(
            nn.Linear(self.lstm_hidden_count*self.window, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1)
        )


    def forward(self, x, hidden_state):
        #Apply feature extraction to every vector in the sequence.





