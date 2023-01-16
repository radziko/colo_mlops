import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from streamlit_image_select import image_select
from torchvision import transforms

from src.models.load_model_artifact import load_model_artifact
from src.models.model import CIFAR10Module
from src.utils.logger import get_logger

# @hydra.main(
#    config_path="../config", config_name="default_config.yaml", version_base="1.2")


@st.experimental_memo
def get_model():
    # print(f"configuration: \n {OmegaConf.to_yaml(config.app)}")
    logger = get_logger({"logger": "wandb"})
    checkpoint_reference = "model-90it9ou2:best_k"

    artifact_dir = load_model_artifact(logger, checkpoint_reference)
    model = CIFAR10Module().load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
    return model


def predict(model, input, categories):
    probabilities = torch.exp(model.forward(input))
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    for id, prob in zip(top5_catid[0], top5_prob[0]):
        # print(id, prob[0])
        st.write(categories[id.item()], round(prob.item(), 3))


def transformation(input_img):
    img = np.asarray(input_img)[:, :, :3]
    img = Image.fromarray(img)
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x.view(-1, 3, 32, 32)
            ),  # Convert it to square images
            transforms.Normalize((0,), (1,)),  # Then standardize
        ]
    )
    return transform(img)


def main():
    st.markdown("# Pick an image from the list to classify")
    st.sidebar.markdown("# Pick an image from the list to classify")
    st.text("Use our model to predict an image you choose.")
    model = get_model()
    model.eval()
    # Categories in cifar10
    categories = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    img = image_select(
        label="Select an image",
        images=[
            Image.open("app/images/car1.jpg"),
            Image.open("app/images/deer1.jpg"),
            Image.open("app/images/dog1.jpg"),
            Image.open("app/images/ship1.jpg"),
            Image.open("app/images/bird1.jpg"),
            Image.open("app/images/cat1.jpg"),
            Image.open("app/images/frog1.jpg"),
            Image.open("app/images/truck1.jpg"),
            Image.open("app/images/airplane1.jpg"),
            Image.open("app/images/horse1.jpg"),
        ],
        captions=[
            "Automobile",
            "Deer",
            "Dog",
            "Ship",
            "Bird",
            "Cat",
            "Frog",
            "Truck",
            "Airplane",
            "Horse",
        ],
    )
    if st.button("Click Here to Classify"):
        my_img = transformation(img)

        with st.spinner("Classifying ..."):
            time.sleep(2)
            st.write("Model predicttions:")
            predict(model, my_img, categories)
            st.success("Done!")


if __name__ == "__main__":
    main()
