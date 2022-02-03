import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class RecoveryRateDataset(Dataset):

    def __init__(self, file_path, transform=None):

        self.raw_dataframe = pd.read_pickle(file_path)
        self.transform = transform


    def _fill_nans(self, raw_dataframe):
        raw_dataframe
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        return NotImplemented

