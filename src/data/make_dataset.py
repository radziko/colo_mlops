# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pickle
import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset


def unpickle(file):
    with open(file, 'rb') as file:
        dict = pickle.load(file, encoding='latin1')
    return dict

def transformer(data):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0,), (1,)),
                                    transforms.Lambda(lambda x: torch.flatten(torch.swapdims(x, 0, 1), start_dim=1)), # Remove flatten to make conv
                                    transforms.Lambda(lambda x: x.to(torch.float32))])
    return transform(data)

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    #Getting training data
    train_data = []
    train_labels = []
    logger.info('Starting loading. Treating all files starting with \"data\" as training data.')
    for filename in os.listdir(input_filepath):
        if filename[:4] == 'data':
            train_data.append(unpickle(f'{input_filepath}/{filename}')['data'])
            train_labels.append(unpickle(f'{input_filepath}/{filename}')['labels'])
    train_data, train_labels = np.concatenate(train_data), np.concatenate(train_labels)

    #Getting test data
    test_data = unpickle(f'{input_filepath}/test_batch')['data']
    test_labels = unpickle(f'{input_filepath}/test_batch')['labels']

    # Transforming data
    train_set = {"data": transformer(train_data), "labels": torch.LongTensor(train_labels)}
    train_set = TensorDataset(train_set['data'], train_set['labels']) 
    test_set = {"data": transformer(test_data), "labels": torch.LongTensor(test_labels)}
    test_set = TensorDataset(test_set['data'], test_set['labels']) 

    torch.save(train_set, f=f'{output_filepath}/train.data')
    torch.save(test_set, f=f'{output_filepath}/test.data')

    logger.info('Finished. Raw data has been processed.')



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
