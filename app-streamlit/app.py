import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.models.load_model_artifact import load_model_artifact
from src.models.model import CIFAR10Module
from src.utils.logger import get_logger

# @hydra.main(
#    config_path="../config", config_name="default_config.yaml", version_base="1.2")


def get_model():
    # print(f"configuration: \n {OmegaConf.to_yaml(config.app)}")

    logger = get_logger({"logger": "wandb"})
    checkpoint_reference = "model-90it9ou2:best_k"

    artifact_dir = load_model_artifact(logger, checkpoint_reference)
    model = CIFAR10Module().load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
    return model


model = get_model()
model.eval()
output = model.forward(torch.rand((1, 3, 32, 32)))
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


def predict(input):

    probabilities = torch.exp(model.forward(input))
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    for id, prob in zip(top5_catid[0], top5_prob[0]):
        # print(id, prob[0])
        st.sidebar.write(categories[id.item()], round(prob.item(), 3))


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
                # .swapdims(1, 3)
                # .swapdims(1, 2)  # Have to swap dimensions twice to avoid transpose
            ),  # Convert it to square images
            transforms.Normalize((0,), (1,)),  # Then standardize
        ]
    )
    return transform(img)


if __name__ == "__main__":

    # Designing the interface
    st.title("Object Classifier - An MLOPS Project")
    # For newline
    st.write("\n")

    image = Image.open("app-streamlit/images/image.png")
    show = st.image(image, use_column_width=True)

    st.sidebar.title("Upload Image")

    # Disabling warning
    st.set_option("deprecation.showfileUploaderEncoding", False)
    # Choose your own image
    uploaded_file = st.sidebar.file_uploader(" ", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:

        u_img = Image.open(uploaded_file)
        show.image(u_img, "Uploaded Image", use_column_width=True)
        # We preprocess the image to fit in algorithm.

        my_image = transformation(u_img)
    # For newline
    st.sidebar.write("\n")

    if st.sidebar.button("Click Here to Classify"):

        if uploaded_file is None:

            st.sidebar.write("Please upload an Image to Classify")

        else:

            with st.spinner("Classifying ..."):

                st.sidebar.header("Algorithm Predicts: ")
                time.sleep(2)
                predict(my_image)

                st.success("Done!")
