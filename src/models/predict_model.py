import multiprocessing
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
from load_model_artifact import load_model_artifact
from omegaconf import OmegaConf

from src.data.cifar10_datamodule import CIFAR10DataModule
from src.models.model import CIFAR10Module
from src.utils.logger import get_logger


@hydra.main(
    config_path="../../config", config_name="default_config.yaml", version_base="1.2"
)
def predict(config):
    ''' Takes a config file and does inference with a trained model on the CIFAR10 test set. '''

    print(f"configuration: \n {OmegaConf.to_yaml(config.testing)}")

    hparams = config.testing
    pl.seed_everything(hparams["seed"])

    logger = get_logger(hparams)

    artifact_dir = load_model_artifact(logger, hparams["ckpt_path"])
    model = CIFAR10Module().load_from_checkpoint(Path(artifact_dir) / "model.ckpt")

    org_cwd = hydra.utils.get_original_cwd()
    data = CIFAR10DataModule(
        data_dir=os.path.join(org_cwd, "data/processed/CIFAR10"),
        batch_size=hparams["batch_size"],
        num_workers=multiprocessing.cpu_count(),
    )

    trainer = pl.Trainer(accelerator=hparams["accelerator"], logger=logger)

    trainer.test(model=model, dataloaders=data)


if __name__ == "__main__":
    predict()
