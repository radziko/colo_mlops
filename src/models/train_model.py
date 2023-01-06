import pytorch_lightning as pl
import timm
from torch import nn, optim

from src.data.cifar10_datamodule import CIFAR10DataModule
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project="mlops_project", log_model=False)


class CIFAR10ViT(pl.LightningModule):
    def __init__(self, classifier: nn.Module):
        super().__init__()
        self.classifier = classifier
        self.loss = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.classifier(x)

        loss = self.loss(y_hat, y)
        self.log('training_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


classifier = timm.create_model("resnet18")

model = CIFAR10ViT(classifier)

trainer = pl.Trainer(logger=wandb_logger, default_root_dir="models/")

trainer.fit(model, CIFAR10DataModule(batch_size=128))
