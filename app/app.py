import io

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.models.model import CIFAR10Module


def predict(image, categories):
    model = CIFAR10Module().load_from_checkpoint("models/model.ckpt")
    model.eval()

    preprocess = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model.forward(input_batch)
        probs = torch.exp(output)

    top5_prob, top5_catid = torch.topk(probs, 5)

    top5_prob = top5_prob.numpy()[0]
    top5_catid = top5_catid.numpy()[0]

    for i in range(len(top5_prob)):
        print(categories[top5_catid[i]], top5_prob[i].item())

        # st.write(categories[top5_catid[i]], top5_prob[i].item())

    return


def load_image():
    uploaded_file = st.file_uploader(label="Pick an image to test")
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def load_image_local(path):
    return Image.open(path)


def load_labels(labels_file):
    with open(labels_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories


def main():
    img = load_image_local("test_img.jpg")
    labels = load_labels("classes.txt")
    predict(image=img, categories=labels)

    # st.title('Custom model demo')
    # model = load_model(MODEL_PATH)
    # categories = load_labels(LABELS_PATH)
    # image = load_image()
    # result = st.button('Run on image')
    # if result:
    #     st.write('Calculating results...')
    #     predict(model, categories, image)


if __name__ == "__main__":
    main()
