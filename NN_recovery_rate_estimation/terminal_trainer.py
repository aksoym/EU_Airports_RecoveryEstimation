import torch

from torch.utils.data import DataLoader
from dataset import RecoveryRateDataset
import os
from training_utils import compute_mean_and_std, apply_std_scaling, set_seed
from model_lightning import LSTMEstimator
from training_loop import model_train

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

SEED = 1234
seed_everything(SEED, workers=True)

import os
os.environ["WANDB_START_METHOD"] = "thread"
import wandb