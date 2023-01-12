import multiprocessing
import os

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.data.cifar10_datamodule import CIFAR10DataModule
from src.models.callbacks.log_validation_predictions_callback import (
    LogValidationPredictionsCallback,
)
from src.models.model import CIFAR10Module, get_model
from src.utils.logger import get_logger


@hydra.main(
    config_path="../../config", config_name="default_config.yaml", version_base="1.2"
)
def train(config):
    ''' Takes a config file and trains the model. Saves a model checkpoint in weights and biases after each epoch. '''

    print(f"configuration: \n {OmegaConf.to_yaml(config.training)}")

    hparams = config.training
    pl.seed_everything(hparams["seed"])
    model = CIFAR10Module(classifier=get_model("resnet18", False), lr=hparams["lr"])

    log_predictions_callback = LogValidationPredictionsCallback()
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/accuracy", mode="max"
    )

    trainer = pl.Trainer(
        accelerator=hparams["accelerator"],
        max_epochs=hparams["n_epochs"],
        auto_lr_find=hparams["auto_lr_find"],
        logger=get_logger(hparams),
        default_root_dir="models/",
        callbacks=[log_predictions_callback, model_checkpoint_callback],
    )

    org_cwd = hydra.utils.get_original_cwd()
    data = CIFAR10DataModule(
        data_dir=os.path.join(org_cwd, "data/processed/CIFAR10"),
        batch_size=hparams["batch_size"],
        num_workers=multiprocessing.cpu_count(),
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    train()
