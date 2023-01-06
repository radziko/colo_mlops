'''
SimplerNetV1 in Pytorch.
The implementation is basded on : 
https://github.com/D-X-Y/ResNeXt-DenseNet
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn, optim
import timm


def get_model(model: str, pretrained:bool=False):
    our_model = timm.create_model(model, pretrained=pretrained)
    return our_model


class CIFAR10ViT(pl.LightningModule):
    def __init__(self, classifier: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.classifier = classifier
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.classifier(x)

        loss = self.loss(y_hat, y)
        self.log('training_loss', loss)

        return loss


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer