import pytorch_lightning as pl
import torch
import hydra
from omegaconf import OmegaConf
from src.models.model import CIFAR10ViT, simplenet
from src.data.cifar10_datamodule import CIFAR10DataModule
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project="mlops_project", log_model=False, entity='team-colo')


@hydra.main(config_path="../../config", config_name='default_config.yaml')
def train(config):
    
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")

    hparams = config.experiment
    pl.seed_everything(hparams['seed'])
    model = CIFAR10ViT(classifier=simplenet(), lr=hparams['lr'])

    trainer = pl.Trainer(
        accelerator=hparams['accelerator'],
        max_epochs=hparams['n_epochs'],
        auto_lr_find=hparams['auto_lr_find'],
        logger=wandb_logger, 
        default_root_dir="models/")

    
    org_cwd = hydra.utils.get_original_cwd()

    data = CIFAR10DataModule(data_dir=org_cwd + '/data/processed/CIFAR10', batch_size=hparams['batch_size'])

    trainer.fit(model, data)

if __name__ == "__main__":
    train()