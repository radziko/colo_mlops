import pytorch_lightning as pl
import timm
from torch import nn, optim
import hydra
from omegaconf import OmegaConf
from src.models.model import simplenet
from src.data.cifar10_datamodule import CIFAR10DataModule

class CIFAR10ViT(pl.LightningModule):
    def __init__(self, classifier: nn.Module):
        super().__init__()
        self.classifier = classifier
        self.loss = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.classifier(x)

        loss = self.loss(y_hat, y)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


classifier = simplenet()

model = CIFAR10ViT(classifier)

trainer = pl.Trainer()

data = CIFAR10DataModule(batch_size=32)

trainer.fit(model, data)