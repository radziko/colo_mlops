import os
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from pytorch_lightning.loggers import Logger, TensorBoardLogger, WandbLogger


def get_logger(config: dict) -> Optional[Logger]:
    """Returns a logger for the model logging.

    Args:
        config: A config file with a "logger" key stating the logger.

    Returns:
        logger: The logger for the model.

    """
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    if config["logger"] == "wandb":
        wandb_project = os.environ.get("WANDB_PROJECT")
        wandb_entity = os.environ.get("WANDB_ENTITY")

        assert wandb_entity is not None
        assert wandb_project is not None

        logger = WandbLogger(
            project=wandb_project,
            log_model="all",
            entity=wandb_entity,
            save_dir="outputs/",
        )
    elif config["logger"] == "tensorboard":
        logger = TensorBoardLogger("outputs", "runs")
    else:
        logger = None

    return logger
