import torch
import torch_optimizer
import pytorch_lightning as pl

from src.model import get_model


## See: https://dacon.io/competitions/official/235870/codeshare/4383
class RestrictedImagenetModel(pl.LightningModule):
    
    def __init__(self):
        super(RestrictedImagenetModel, self).__init__()
        self.model = get_model(pretrained=False) ## ResNet-50
        self.criterion = torch.nn.CrossEntropyLoss()


    def _accuracy(self, y_pred, y_true):
        y_true = y_true.cpu()
        y_pred = torch.argmax(y_pred, dim=1).cpu()
        return (y_pred == y_true).to(torch.float).mean()


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
        acc = self._accuracy(y_pred=y_hat, y_true=y)

        ## Return loss.
        return loss, acc


    def training_step(self, train_batch, batch_idx):
        ## Step it.
        loss, acc = self.step(train_batch)

        ## Record log.
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, logger=True)

        ## Return values to either update gradient and accumulate on progress bar.
        return {"loss": loss, "acc": acc}


    def validation_step(self, val_batch, batch_idx):
        ## Step it.
        loss, acc = self.step(val_batch)

        ## Record log.
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)

        ## Return values to either update gradient and accumulate on progress bar.
        return {"val_loss": loss, "val_acc": acc}
