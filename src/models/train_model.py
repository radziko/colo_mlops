import multiprocessing
import os
from typing import Optional

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import Logger, TensorBoardLogger, WandbLogger

from src.data.cifar10_datamodule import CIFAR10DataModule
from src.models.model import CIFAR10Model, get_model


from pytorch_lightning.callbacks import Callback
 

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


class LogPredictionsCallback(Callback):
    
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
 
        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case
        
        # Let's log 10 sample image predictions from first batch
        if batch_idx == 0:
            n = 10
            x, y = batch
            images = [img for img in x[:n]]
            print('yyyyyyyy',y.shape)
            print('outputs',outputs.shape)
            captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' for y_i, y_pred in zip(y[:n], outputs[:n])]
            
            # Option 1: log images with `WandbLogger.log_image`
            trainer.logger.log_image(key='sample_images', images=images, caption=captions)


            # Option 2: log predictions as a Table
            """
            columns = ['image', 'ground truth', 'prediction']
            data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
            wandb_logger.log_table(key='sample_table', columns=columns, data=data)
            """


@hydra.main(
    config_path="../../config", config_name="default_config.yaml", version_base="1.2"
)
def train(config):
    print(f"configuration: \n {OmegaConf.to_yaml(config.training)}")

    hparams = config.training
    pl.seed_everything(hparams["seed"])
    model = CIFAR10Model(classifier=get_model("resnet18", False), lr=hparams["lr"])

    log_predictions_callback = LogPredictionsCallback()

    trainer = pl.Trainer(
        accelerator=hparams["accelerator"],
        max_epochs=hparams["n_epochs"],
        auto_lr_find=hparams["auto_lr_find"],
        logger=get_logger(hparams),
        default_root_dir="models/",
        callbacks=[log_predictions_callback, pl.callbacks.ModelCheckpoint(monitor="val/accuracy", mode="max")]
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