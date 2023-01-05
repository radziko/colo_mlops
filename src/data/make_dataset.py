# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split
from torch import Generator
import torch


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("-s", "--seed", type=int, default=42)
def main(input_filepath, output_filepath, seed):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    transform = transforms.Compose([transforms.ToTensor()])

    training_data = CIFAR10(
        root=os.path.join(input_filepath, "CIFAR10"),
        train=True,
        download=True,
        transform=transform,
    )
    test_data = CIFAR10(
        root=os.path.join(input_filepath, "CIFAR10"),
        train=False,
        download=True,
        transform=transform,
    )
    # Create validation set from training data
    training_data, validation_data = random_split(training_data, [0.95, 0.05], generator=Generator().manual_seed(seed))

    output_dir = os.path.join(output_filepath, "CIFAR10",)
    os.makedirs(output_dir, exist_ok=True)

    torch.save(training_data, os.path.join(output_dir, "train.pt"))
    torch.save(validation_data, os.path.join(output_filepath, "CIFAR10", "validation.pt"))
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
