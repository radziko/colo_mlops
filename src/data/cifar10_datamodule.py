import os

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


class CIFAR10DataModule(pl.LightningDataModule):
    cifar10_train: TensorDataset
    cifar10_validation: TensorDataset
    cifar10_test: TensorDataset

    def __init__(self, data_dir: str = "data/processed/CIFAR10", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        train_dataset = torch.load(os.path.join(self.data_dir, "train.pt"))
        validation_dataset = torch.load(os.path.join(self.data_dir, "validation.pt"))
        test_dataset = torch.load(os.path.join(self.data_dir, "test.pt"))

        self.cifar10_train = TensorDataset(
            train_dataset["images"], train_dataset["labels"]
        )
        self.cifar10_validation = TensorDataset(
            validation_dataset["images"], validation_dataset["labels"]
        )
        self.cifar10_test = TensorDataset(
            test_dataset["images"], test_dataset["labels"]
        )

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.cifar10_validation, batch_size=self.batch_size, num_workers=12)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, num_workers=12)

    def predict_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, num_workers=12)