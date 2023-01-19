import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


class CIFAR10DataModule(pl.LightningDataModule):
    """Data module for loading the CIFAR10 dataset.

    Attributes:
        cifar10_train: TensorDataset, the train set of CIFAR10 dataset
        cifar10_validation: TensorDataset, the validation set of CIFAR10 dataset
        cifar10_test: TensorDataset, the test set of CIFAR10 dataset
        num_workers: int, number of worker to use for data loading
    """

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
        """Initializes the data module.

        Args:
            data_dir: str, directory where the CIFAR10 dataset is stored.
            batch_size: int, size of the mini-batch.
            num_workers: int, number of worker to use for data loading
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        """Loads the CIFAR10 dataset from files.

        Args:
            stage: str, the stage for which the setup is being run (e.g. 'fit', 'test')
        """
        train_dataset = torch.load(os.path.join(self.data_dir, "train.pt"))
        validation_dataset = torch.load(os.path.join(self.data_dir, "validation.pt"))
        test_dataset = torch.load(os.path.join(self.data_dir, "test.pt"))

        self.cifar10_train = train_dataset
        self.cifar10_validation = validation_dataset
        self.cifar10_test = test_dataset

    def train_dataloader(self) -> DataLoader:
        """Returns the dataloader for the validation set.

        Returns:
            DataLoader, the dataloader for the validation set.
        """
        return DataLoader(
            self.cifar10_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the dataloader for the validation set.

        Returns:
            DataLoader, the dataloader for the validation set.
        """
        return DataLoader(
            self.cifar10_validation,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the dataloader for the test set.

        Returns:
            DataLoader, the dataloader for the test set.
        """
        return DataLoader(
            self.cifar10_test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self) -> DataLoader:
        """Returns the dataloader for the predict (=test) set.

        Returns:
            DataLoader, the dataloader for the predict (=test) set.
        """
        return DataLoader(
            self.cifar10_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
