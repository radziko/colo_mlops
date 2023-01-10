import pytest
from src.models.model import CIFAR10Model, get_model
from tests import _PATH_DATA
import os
import torch

def test_model_output():
    model = CIFAR10Model(classifier=get_model("resnet18", False))

    x = torch.randint(0, 1000, (3,32,32))
    y = torch.randint(0, 1000, (10,1))

    print(model.forward(x,y))
if __name__ == '__main__':
    test_model_output()