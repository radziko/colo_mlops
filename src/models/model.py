"""
SimplerNetV1 in Pytorch.
The implementation is basded on :
https://github.com/D-X-Y/ResNeXt-DenseNet
"""

from typing import Tuple

import pytorch_lightning as pl
import timm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchmetrics import AUROC, Accuracy, F1Score, MetricCollection, Precision, Recall


def get_model(model: str, pretrained: bool = False):
    our_model = timm.create_model(model, pretrained=pretrained, num_classes=10)
    return our_model


class CIFAR10Module(pl.LightningModule):
    def __init__(
        self, classifier: nn.Module = get_model("resnet18", False), lr: float = 1e-3
    ):
        super().__init__()
        self.classifier = classifier
        self.loss = nn.NLLLoss()
        self.lr = lr
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
                "val/f1": F1Score(task="multiclass", num_classes=10),
                "val/precision": Precision(task="multiclass", num_classes=10),
                "val/recall": Recall(task="multiclass", num_classes=10),
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
        self.log("training_loss", loss)
        self.log_dict(self.train_metrics.compute(), on_step=True, on_epoch=True)

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
        self.log("validation_loss", loss)
        self.log_dict(self.validation_metrics.compute(), on_step=False, on_epoch=True)

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
        self.log("test_loss", loss)
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True)

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
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
