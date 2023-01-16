import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


class CIFAR10DataModule(pl.LightningDataModule):
    cifar10_train: TensorDataset
    cifar10_validation: TensorDataset
    cifar10_test: TensorDataset
    num_workers: int

    def __init__(
        self,
        data_dir: str = "data/processed/CIFAR10",
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        train_dataset = torch.load(os.path.join(self.data_dir, "train.pt"))
        validation_dataset = torch.load(os.path.join(self.data_dir, "validation.pt"))
        test_dataset = torch.load(os.path.join(self.data_dir, "test.pt"))

        self.cifar10_train = train_dataset
        self.cifar10_validation = validation_dataset
        self.cifar10_test = test_dataset

    def train_dataloader(self):
        """Returns the dataloader for the train set."""
        return DataLoader(
            self.cifar10_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        """Returns the dataloader for the validation set."""
        return DataLoader(
            self.cifar10_validation,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Returns the dataloader for the test set."""
        return DataLoader(
            self.cifar10_test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        """Returns the dataloader for the predict (=test) set."""
        return DataLoader(
            self.cifar10_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
