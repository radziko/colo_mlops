import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

import wandb


class LogValidationPredictionsCallback(Callback):
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ):
        """Called when the validation batch ends.

        Args:
            trainer: The pytorch lightning trainer.
            pl_module: The pytorch lightning module.
            outputs: The outputs of the model.
            batch: A tuple of the input and label.
            batch_idx: The current batch index.
            dataloader_idx: The current dataloader index.
        """

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 10 sample image predictions from first batch
        if batch_idx == 0:
            n = 10
            x, y = batch

            columns = ["image", "ground truth", "prediction"]
            data = [
                [wandb.Image(x_i), y_i, y_pred]
                for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))
            ]
            trainer.logger.log_table(
                key="validation_samples", columns=columns, data=data
            )
