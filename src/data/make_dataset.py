# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import torch
import torchvision
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import random_split
from torchvision import transforms


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("-s", "--seed", type=int, default=42)
def main(input_filepath: str, output_filepath: str, seed: int):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).

    Args:
        - input_filepath: Filepath for the raw data set.
        - output_filepath: Filepath for where to save the processed data set.
        - seed: Seed for the data set split to train/test.
    """

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=input_filepath, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=input_filepath, train=False, download=True, transform=transform_test
    )

    train_data, validation_data = random_split(
        trainset, [45000, 5000], generator=torch.Generator().manual_seed(seed)
    )
    test_data = testset

    output_dir = os.path.join(output_filepath, "CIFAR10")
    os.makedirs(output_dir, exist_ok=True)

    torch.save(train_data, os.path.join(output_dir, "train.pt"))
    torch.save(
        validation_data, os.path.join(output_filepath, "CIFAR10", "validation.pt")
    )
    torch.save(test_data, os.path.join(output_filepath, "CIFAR10", "test.pt"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
