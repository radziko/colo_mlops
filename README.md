colo_mlops
==============================
Carl Anton Schmidt

Joachim Schrøder Andersson

Jonas Hoffmann

Julius Radzikowski


## Project Description for Week 1:

#### Overall goal of the project
The overall goal of the project is to use a deep learning classification model, to classify images from the CIFAR-10 dataset into 10 classes, which include "dog", "truck", "ship" amongst others. Another main goal of the project is to deploy as many of the tools from the course as possible, to make this project easy to understand and reproduce.

#### What framework are you going to use (PyTorch Image Models, Transformer, Pytorch-Geometrics)
This project will deploy the framework [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models), since we will be working with image classification.

#### How do you intend to include the framework into your project
The framework will be used to obtain a pretrained version of the model Resnet18, which will form the base of doing transfer learning; doing further fine-tuning and then evaluation of the model. Other versions of ResNet might be tested, also from TIMM (Pytorch Image Models).

#### What data are you going to run on (initially, may change)
Initially, the data to be used is the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The dataset consists of 60000 images distributed equally between 10 classes. An image is 32x32 and contains color. This dataset was chosen due to its simplicity for use in image classification since the overarching goal of this project is to deploy MLOps tools.

#### What deep learning models do you expect to use
From the before-mentioned model-framework, the RestNet18 model ([Documentation](https://arxiv.org/abs/1512.03385)) is planned to be used. We might try out additional models if time allow us to do so.


MLOps project

## Project setup
First, install all the ```pip``` requirements:
```bash
pip install -r requirements.txt
```
Or:
```bash
make requirements
```
Second, install the _pre-commit_ hooks:
```bash
pre-commit install
```
And then download the data with ```dvc```:
```bash
dvc pull
```
Finally, copy the ```.env.default``` to ```.env``` and fill out the environment variables.

## Commands to use:
1. Install requirements:        ```pip install -r requirements.txt```
2. Install _pre-commit_ hooks:  ```pre-commit install```
3. Load data from dvc:          ```dvc pull```
4. Set up the environment:      This requires a bit more brainpower, since the user have to manually look up these values. The above commands should have created a file named ```.env.default```. First rename it ```.env```. Second fill out the values. In order to do so, one need a [Wandb](https://wandb.ai/home)-account with a project. The ```WANDB_MODELCHECKPOINT``` entity should have a value like _model-90it9ou2:best_k_.
5. Use dataprocessor:           ```make data``` OR ```python src/data/make_dataset.py data/raw data/processed```
6. Train model:                 ```python src/models/train_model.py```
7. Test model:                  ```python src/models/predict_model.py```
8. Run streamlit local:         ```streamlit run  app/upload.py```



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
