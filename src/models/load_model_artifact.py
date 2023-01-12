import os

from dotenv import find_dotenv, load_dotenv
from pytorch_lightning.loggers import WandbLogger


def load_model_artifact(logger: WandbLogger, checkpoint_reference: str) -> str:
    """Downloads a model checkpoint from weights and biases
    and saves is it in the "models/" directory."""

    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    wandb_project = os.environ.get("WANDB_PROJECT")
    wandb_entity = os.environ.get("WANDB_ENTITY")

    assert wandb_entity is not None
    assert wandb_project is not None

    artifact_ref = f"{wandb_entity}/{wandb_project}/{checkpoint_reference}"
    return logger.download_artifact(
        artifact=artifact_ref,
        save_dir="models/",
        artifact_type="model",
        use_artifact=True,
    )
