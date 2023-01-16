import os
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
#   config_path="../config", config_name="default_config.yaml", version_base="1.2")


@st.experimental_memo
def get_model():
    # print(f"configuration: \n {OmegaConf.to_yaml(config.app)}")
    logger = get_logger({"logger": "wandb"})
    # checkpoint_reference = "model-90it9ou2:best_k"
    checkpoint_reference = os.environ.get("WANDB_MODELCHECKPOINT")

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
    st.markdown("# Upload an image to classify")
    st.sidebar.markdown("# Classify your own image")
    st.text("Use our model to predict an image of your own.")

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

    # Disabling warning
    st.set_option("deprecation.showfileUploaderEncoding", False)
    # Choose your own image
    uploaded_file = st.file_uploader(" ", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        u_img = Image.open(uploaded_file)
        st.image(u_img, "Uploaded Image", use_column_width=True)
        # We preprocess the image to fit in algorithm.
        my_image = transformation(u_img)

    st.sidebar.write("\n")

    if st.button("Click Here to Classify"):
        if uploaded_file is None:
            st.write("Please upload an Image to Classify")

        else:
            with st.spinner("Classifying ..."):
                st.header("Algorithm Predicts: ")
                time.sleep(2)
                predict(model, my_image, categories)

                st.success("Done!")


if __name__ == "__main__":
    main()
