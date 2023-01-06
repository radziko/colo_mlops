colo_mlops
==============================

## Project Description for Week 1:

#### Overall goal of the project
The overall goal of the project is to use a classification model to classify images into 10 classes, which include "dog", "truck", "ship" etc. 

#### What framework are you going to use (PyTorch Image Models, Transformer, Pytorch-Geometrics)
Unfortunately, this project will not deploy 

#### How do you intend to include the framework into your project
For a start, the framework will be used to obtain atleast one pretrained model, which will form the base of doing transfer learning; doing further training and evaluation of the model.

#### What data are you going to run on (initially, may change)
Initially, the data to be used is the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The dataset consists of 60000 images distributed equally between 10 classes. An image is 32x32 and contains color. 

#### What deep learning models do you expect to use
From the before-mentioned model-framework, the RestNet model ([Documentation](https://arxiv.org/abs/1512.03385)) is planned to be used.


MLOps project
## Commands to use:
1. Load data from dvc:  "pull dvc"
2. Use dataprocessor:   "make data" OR "python src/data/make_dataset.py data/raw data/processed"


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
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
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
