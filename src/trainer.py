import torch
import torch_optimizer

import pytorch_lightning as pl
import torchmetrics

from src.model import get_model


## See: https://dacon.io/competitions/official/235870/codeshare/4383
class RestrictedImagenetModel(pl.LightningModule):
    
    def __init__(self):
        super(RestrictedImagenetModel, self).__init__()
        self.model = get_model(pretrained=False) ## ResNet-50

        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy()
        self.valid_accuracy = torchmetrics.Accuracy()


    def forward(self, x):
        return self.model(x)


    def configure_optimizers(self):
        return torch_optimizer.RAdam(self.parameters(), lr=1e-4)


    def step(self, batch):
        ## Unpack.
        x = batch["inp"]
        y = batch["tar"]

        ## Forward.
        y_hat = self(x)

        ## Calculate loss without softmax (calculated by class indices).
        loss = self.criterion(y_hat, y) ## input & target

        ## Return loss.
        return loss, y_hat, y


    def training_step(self, train_batch, batch_idx):
        ## Step it.
        loss, y_hat, y = self.step(train_batch)
        self.train_accuracy(y_hat, y)

        ## Record log.
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)

        ## Just return loss.
        return loss


    def validation_step(self, val_batch, batch_idx):
        ## Step it.
        loss, y_hat, y = self.step(val_batch)
        self.valid_accuracy(y_hat, y)

        ## Record log.
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.valid_accuracy, on_step=True, on_epoch=True, prog_bar=True)
