# -*- coding: utf-8 -*-
import logging
import os
import pickle
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from torch import Generator
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10


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

    # Load training data
    training_datasets = []

    for i in range(1, 6):
        data_path = os.path.join(input_filepath, "CIFAR10", f"data_batch_{i}")

        with open(data_path, "rb") as f:
            training_datasets.append(pickle.load(f, encoding="bytes"))

    training_data = np.concatenate([ds[b"data"] for ds in training_datasets])

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x.view(-1, 3, 32, 32)
            ),  # Convert it to square images
            transforms.Normalize((0,), (1,)),  # Then standardize
        ]
    )

    training_images = transform(training_data)
    training_labels = torch.LongTensor(
        np.concatenate([ds[b"labels"] for ds in training_datasets])
    )

    # Split training set into training and validation
    (
        training_images,
        validation_images,
        training_labels,
        validation_labels,
    ) = train_test_split(
        training_images, training_labels, test_size=0.05, random_state=seed
    )

    # Load test data
    with open(os.path.join(input_filepath, "CIFAR10", "test_batch"), "rb") as f:
        test_dataset = pickle.load(f, encoding="bytes")

    test_images = transform(test_dataset[b"data"])
    test_labels = torch.LongTensor(test_dataset[b"labels"])

    train_data = {"images": training_images, "labels": training_labels}
    validation_data = {"images": validation_images, "labels": validation_labels}
    test_data = {"images": test_images, "labels": test_labels}

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
