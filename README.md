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

## Project Quick Start
1. Before you can do anything with this project, you need to clone the repository. This can be done with
```bash
git clone git@github.com:radziko/colo_mlops.git
```
2. We recommend that you setup a new ```Conda``` environment for this project. Do this with the command
```bash
make create_environment
```
3. Now that you have the code locally, you need install all the ```pip``` requirements:
```bash
make requirements
```
4. Install the _pre-commit_ hooks:
```bash
pre-commit install
```
5. Download the data with ```dvc```:
```bash
pip install dvc
pip install 'dvc[gs]'
dvc pull
```
6. Finally, many of the scripts require environment variables. The above commands should have created a file named ```.env.default```. Copy this, and rename the copy as ```.env```. In order to do so, one needs a [Wandb](https://wandb.ai/home)-account to generate an API-key ```WANDB_API_KEY```. Second, you should create a team within Wandb that accounts for the ```WANDB_ENTITY```, and within that team a project that becomes the ```WANDB_PROJECT```. The ```WANDB_MODELCHECKPOINT``` entity should have a value like _model-90it9ou2:best_k_, and will only be created once a training run has been completed. Note, there are also the environment variables ```SERVICE_ACCOUNT``` and ```GCP_PROJECT```, however, we will fill these out later when needed.

Now, you should have all the code locally, and an environment that is able to run the project.

## Key Commands to use
1. To process the raw data, you should use the command
```bash
make data
```
2. To train the model use
```bash
make train
```
- Note, the training is logged in your wandb account.

3. To evaluate your newly trained model you can simply do
```bash
make evaluate
```
- Before you do this, you will need to find a model checkpoint within Wandb that you must enter in the ```.env```-file under the ```WANDB_MODELCHECKPOINT``` variable. This is found in Wandb under _"Homepage -> 'Your Team' -> Projects -> 'Your Project' -> 'Latest Run' -> Artifacts"_. Here, you should see a list of "_Output artifacts_", and in this there is the type "_model_". Copy the model name and insert this as the model checkpoint. If you wish to test the best model, you should put "_:best_k_" instead of e.g. "_:v_3_".

4. You now have a trained model saved in Wandb that you can easily evaluate. This model is implement in an app that predicts on uploaded images. To host this app locally, you should use the command
```bash
make app
```
- This will host the app under ```http://localhost:8501/```.

This covers the essential commands to use in this project locally. The next section will cover some commands for ```docker```, and ```GCP``` if you have this set up.

## Extra Commands for Docker and GCP
Beforre you continue with this section you will need to fill out the rest of the environment variables in the ```.env```-file. You should have a project within ```GCP``` that has a project-id, and this is the ```GCP_PROJECT```. Second, you should fill out your e-mail that you use in ```GCP```as the ```SERVICE_ACCOUNT``` variables.

1. Given that you have Docker installed, you can containerize numerous element from this project quickly with the command
```bash
make docker_build
```
This will build 3 docker images: _train_, _predict_ and _app_. Also, these images will be tagged according to your ```GCP``` Container Registry in your project.

2. To push the images into your ```GCP```-project you should use the command
```bash
make docker_push
```
3. If you wish to run the training locally you can do
```bash
make docker_train
```
- If you prefer to run the training in Vertex AI, instead do
```bash
make docker_train_cloud
```

4. The same can be done for the predict docker image. Either do
```bash
make docker_predict
```
- To run it locally. For cloud, do
```bash
make docker_predict_cloud
```

5. Finally, you can deploy the app from the dockerr container. To deploy this locally, do
```bash
make docker_deploy_app_local
```
- This will host the app under ```http://localhost:8501/```. If you wish to deploy the app using ```Cloud Run``` you can do
```bash
make docker_deploy_app_cloud
```
- This will deploy the app such that others can use the site, and this can be accessed from a link generated automatically.

This concludes the guide on how to best setup and utilise this project!

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
