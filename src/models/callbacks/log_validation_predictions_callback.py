from pytorch_lightning.callbacks import Callback

import wandb


class LogValidationPredictionsCallback(Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """ Called when the validation batch ends. """

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
