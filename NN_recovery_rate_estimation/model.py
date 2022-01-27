import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


class LSTMEstimator(nn.Module):
    super(LSTMEstimator, self).__init__()

    def __init__(self, lstm_layer_count, parameter_multiplier, input_size):
        self.layer_count = lstm_layer_count
        self.multiplier = parameter_multiplier
        self.input_size = input_size


        self.lstm_layer = nn.LSTM(
            self.input_size
        )

