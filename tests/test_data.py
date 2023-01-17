import os

import pytest

from src.data.cifar10_datamodule import CIFAR10DataModule
from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data_set_length():
    data = CIFAR10DataModule(data_dir=_PATH_DATA)
    data.setup("train")
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()

    assert len(train_loader.dataset) == 45000
    assert len(test_loader.dataset) == 10000
    assert len(val_loader.dataset) == 5000


if __name__ == "__main__":
    test_data_set_length()
