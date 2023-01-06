import pytorch_lightning as pl
import timm
from torch import nn, optim
import hydra
from omegaconf import OmegaConf
from src.models.model import simplenet
from src.data.cifar10_datamodule import CIFAR10DataModule
import os

class CIFAR10ViT(pl.LightningModule):
    def __init__(self, classifier: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.classifier = classifier
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.classifier(x)

        loss = self.loss(y_hat, y)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

@hydra.main(config_path="../../config", config_name='default_config.yaml')
def train(config):
    
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")

    hparams = config.experiment
    pl.seed_everything(hparams['seed'])
    classifier = simplenet()
    model = CIFAR10ViT(classifier)

    trainer = pl.Trainer(
        accelerator=hparams['accelerator'],
        max_epochs=hparams['n_epochs'],
        auto_lr_find=hparams['auto_lr_find'])

    org_cwd = hydra.utils.get_original_cwd()

    data = CIFAR10DataModule(data_dir=org_cwd + '/data/processed/CIFAR10', batch_size=hparams['batch_size'])

    trainer.fit(model, data)

if __name__ == "__main__":
    train()