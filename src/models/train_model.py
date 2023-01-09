import multiprocessing
import os
from typing import Optional

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import Logger, TensorBoardLogger, WandbLogger

from src.data.cifar10_datamodule import CIFAR10DataModule
from src.models.model import CIFAR10Model, get_model


def get_logger(config: dict) -> Optional[Logger]:
    if config["logger"] == "wandb":
        logger = WandbLogger(
            project="mlops_project",
            log_model="all",
            entity="team-colo",
            save_dir="outputs/wandb/",
            prefix="train"
        )
    elif config["logger"] == "tensorboard":
        logger = TensorBoardLogger("outputs", "runs")
    else:
        logger = None

    return logger


@hydra.main(
    config_path="../../config", config_name="default_config.yaml", version_base="1.2"
)
def train(config):
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")

    hparams = config.training
    pl.seed_everything(hparams["seed"])
    model = CIFAR10Model(classifier=get_model("resnet18", False), lr=hparams["lr"])

    trainer = pl.Trainer(
        accelerator=hparams["accelerator"],
        max_epochs=hparams["n_epochs"],
        auto_lr_find=hparams["auto_lr_find"],
        logger=get_logger(hparams),
        default_root_dir="models/",
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val/accuracy", mode="max")]
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
