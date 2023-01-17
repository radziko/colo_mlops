"""
SimplerNetV1 in Pytorch.
The implementation is basded on :
https://github.com/D-X-Y/ResNeXt-DenseNet
"""

from typing import Tuple

import pytorch_lightning as pl
import torch
import torchvision.models
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics import AUROC, Accuracy, F1Score, MetricCollection, Precision, Recall


def create_model():
    model = torchvision.models.resnet18(weights=None, num_classes=10)

    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = nn.Identity()

    return model


class CIFAR10Module(pl.LightningModule):
    def __init__(
        self,
        classifier: nn.Module = create_model(),
        lr: float = 1e-3,
        batch_size: int = 64,
    ):
        super().__init__()
        self.classifier = classifier
        self.loss = nn.NLLLoss()
        self.lr = lr
        self.batch_size = batch_size
        self.save_hyperparameters(ignore=["classifier"])

        self.train_metrics = MetricCollection(
            {
                "train/accuracy": Accuracy(task="multiclass", num_classes=10),
                "train/auroc": AUROC(task="multiclass", num_classes=10),
            }
        )

        self.validation_metrics = MetricCollection(
            {
                "val/accuracy": Accuracy(task="multiclass", num_classes=10),
                "val/auroc": AUROC(task="multiclass", num_classes=10),
            }
        )

        self.test_metrics = MetricCollection(
            {
                "test/accuracy": Accuracy(task="multiclass", num_classes=10),
                "test/auroc": AUROC(task="multiclass", num_classes=10),
                "test/f1": F1Score(task="multiclass", num_classes=10),
                "test/precision": Precision(task="multiclass", num_classes=10),
                "test/recall": Recall(task="multiclass", num_classes=10),
            }
        )

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx
    ) -> torch.Tensor:
        """Calculates the loss for a batch given to the model.

        Args:
            batch: A batch of a predetermined batch size from the CIFAR10 dataset.

        Returns:
            loss: The PyTorch NLLLoss on the batch.
        """
        x, y = batch

        y_hat = self(x)

        self.train_metrics.update(y_hat, y)

        loss = self.loss(y_hat, y)
        self.log("train/loss", loss)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx
    ) -> list[int]:
        """Calculates the predicted labels for a batch given during validation.

        Args:
            batch: A batch from the CIFAR10 dataset.

        Returns:
            pred_label: The predicted label of each data point on the batch,
            i.e. 64 predicted labels if batch_size=64.
        """
        x, y = batch

        y_hat = self(x)

        self.validation_metrics.update(y_hat, y)

        loss = self.loss(y_hat, y)
        self.log("val/loss", loss)
        self.log_dict(self.validation_metrics, on_step=False, on_epoch=True)

        pred_label = torch.argmax(y_hat, dim=1)
        return pred_label

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx
    ) -> list[int]:
        """Calculates the predicted labels for a batch given during testing.

        Args:
            batch: A batch from the CIFAR10 dataset.

        Returns:
            pred_label: The predicted label of each data point on the batch,
            i.e. 64 predicted labels if batch_size=64.
        """
        x, y = batch

        y_hat = self(x)

        self.test_metrics.update(y_hat, y)

        loss = self.loss(y_hat, y)
        self.log("test/loss", loss)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

        pred_label = torch.argmax(y_hat, dim=1)
        return pred_label

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds a layer of log softmax to the forward pass
        to calculate the log-probabilites from NLLLoss.

        Args:
            batch: A batch from the CIFAR10 dataset.

        Returns:
            pred_label: The log-probabilites for a batch.
        """
        return F.log_softmax(self.classifier(x), dim=1)

    def configure_optimizers(self):
        """Configures the optimizer used during training.

        Returns:
            optimizer: The optimizer for training,
            which in this case is the ADAM optimizer.
        """
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = int(round(45000 / self.hparams.batch_size))
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
