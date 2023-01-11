import multiprocessing
import os
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import Logger, TensorBoardLogger, WandbLogger

import wandb
from src.data.cifar10_datamodule import CIFAR10DataModule
from src.models.model import CIFAR10Model


def get_logger(config: dict) -> Optional[Logger]:
    if config["logger"] == "wandb":
        logger = WandbLogger(
            project="mlops_project", log_model=False, entity="team-colo"
        )
    elif config["logger"] == "tensorboard":
        logger = TensorBoardLogger("outputs", "runs")
    else:
        logger = None

    return logger


@hydra.main(
    config_path="../../config", config_name="default_config.yaml", version_base="1.2"
)
def predict(config):
    print(f"configuration: \n {OmegaConf.to_yaml(config.testing)}")

    run = wandb.init(entity="team-colo", project="mlops_project")

    hparams = config.testing
    pl.seed_everything(hparams["seed"])

    checkpoint_reference = "team-colo/mlops_project/" + hparams["ckpt_path"]
    artifact = run.use_artifact(checkpoint_reference, type="model")
    artifact_dir = artifact.download(root="models")

    model = CIFAR10Model().load_from_checkpoint(Path(artifact_dir) / "model.ckpt")

    org_cwd = hydra.utils.get_original_cwd()
    data = CIFAR10DataModule(
        data_dir=os.path.join(org_cwd, "data/processed/CIFAR10"),
        batch_size=hparams["batch_size"],
        num_workers=multiprocessing.cpu_count(),
    )

    trainer = pl.Trainer(
        accelerator=hparams["accelerator"],
    )

    trainer.test(model=model, dataloaders=data)


if __name__ == "__main__":
    predict()
