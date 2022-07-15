import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .model_retrieval import ALBEF
from ..optim import create_optimizer
from ..scheduler import WarmupCosineAnnealingLR


class ALBEFModule(pl.LightningModule):
    def __init__(self,
                 alpha,
                 warm_up,
                 model_kwargs,
                 optimizer_kwargs,
                 lr_scheduler_kwargs,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.model = type(self).__name__

        self.model = ALBEF.from_cktp(model_kwargs)

    def training_step(self, train_batch, batch_idx):
        image, text, idx = train_batch

        if self.current_epoch > 0 or not self.hparams.warm_up:
            alpha = self.hparams.alpha
        else:
            alpha = self.hparams.alpha * min(1, batch_idx / self.trainer.num_training_batches)

        loss_ita, loss_itm, matching_score = self.model(image, text, alpha=alpha, idx=idx)
        loss = loss_ita + loss_itm
        matching_acc = (matching_score >= 0.5).sum() / image.shape[0]

        self.log_dict(dict(loss=loss_itm+loss_ita, loss_itm=loss_itm, loss_ita=loss_ita,
                           train_matching_acc=matching_acc), sync_dist=True, on_epoch=True)
        return loss

    # def on_train_epoch_end(self) -> None:
    #     self.trainer.train_dataloader.loaders.dataset.setup()

    def validation_step(self, val_batch, batch_idx):
        image, text, idx = val_batch
        loss_ita, loss_itm, matching_score = self.model(image, text, alpha=self.hparams.alpha, idx=idx)
        matching_acc = (matching_score >= 0.5).sum() / image.shape[0]

        self.log_dict(
            {'val_loss': loss_itm + loss_ita, 'val_loss_itm': loss_itm, 'val_loss_ita': loss_ita,
             f'val_matching_acc': matching_acc},
            prog_bar=True,
            sync_dist=True)

    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams.optimizer_kwargs, self.model)

        warmup_steps = int(self.hparams.lr_scheduler_kwargs.pop('warmup_epochs') * (
                self.estimated_stepping_batches // self.trainer.max_epochs))
        scheduler = {
            "scheduler": WarmupCosineAnnealingLR(optimizer,
                                                 warmup_steps=warmup_steps,
                                                 max_steps=self.estimated_stepping_batches,
                                                 **self.hparams.lr_scheduler_kwargs),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    @property
    def estimated_stepping_batches(self):
        effective_accum = self.trainer.accumulate_grad_batches * self.trainer.num_devices
        batches = len(self.trainer.datamodule.train_dataloader())
        return (batches // effective_accum) * self.trainer.max_epochs
