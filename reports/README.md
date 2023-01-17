---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

Group number: 16

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s204096, s204162, s204071, s204154.

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

Multiple frameworks were used for the project. The cookiecutter repo-format is used to help the reproducibility of the project. This project will deploy the framework [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) in order to obtain a classification model. Hereunder, the [Pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/stable/) framework is used for avoiding boilerplate code and functionality for model training, etc.

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development enviroment, one would have to run the following commands*
>
> Answer:

We managed dependencies with the pipreqs-package. Using the pipreqs-command in the repository generates a requirements textfile. This includes the packages and their version which are necessary to get the full outcome of this project. For a new user to obtain the dependencies, one will need to run ```pip install -r requirements.txt``` in the terminal of the repo (preferably with a new virtual environment, for example created with conda).

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

In the cookiecutter template, the group has filled out the folders: _models_, _reports_, _src_ (not /features/) and _test_. The folders _references_, _docs_ and _notebooks_. are removed. Multiple new folders are made:  _.dvc_ handles data versioning, _outputs_ contains the output from training and most profound is the folder _app_. This contains images and code to run the streamlit app.

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

We used Black to format our code such that it is PEP 8 compliant, which Flake8 then checks. We also used isort to handle imports properly. It's very important to keep a consistent code style throughout the project.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement?**
>
> Answer:

We have five tests, split in model tests and data tests.

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> **Answer length: 100-200 words.**
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

We have 28% coverage, with our tests. We are only testing core modules, as the data and the model right now. We don't fully expect our code to be error free, as there are always edge-cases, even if we got 100% coverage. It is also hard to attain 100%. This is though a good indicator if your tests are actually doing something.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We used both branches and pull-requests. All features were added as feature branches, as the main branch is protected so you cannot push into it directly. This is much nicer when you are 4 people working together, as everyone can contribute on their own branch.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We did use DVC for controlling our data workflow. It was definetely easier to ensure that everyone had the same version of the dataset across all their multiple devices/distros. It is really easy to just `dvc pull`, and because we set it up in a bucket, it was accessible anywhere.
### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

We used a couple of CI integrations. First we used pre-commit to make sure our CI pipelines didn't fail. We then use four files: Flake8 for formatting, isort for import statements, a tests file running our tests, and finally a coverage file running a coverage report on every PR. An example of the workflow can be seen here: <https://github.com/radziko/colo_mlops/actions/runs/3938966967/jobs/6738273112>

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We used hydra to configure our experiments, such that it provided all the variables for training and testing. We would start it like this:
`python src/models/train_model.py training=default_train`

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We primarily used Wandb to store hyperparameters for each run, and experiments. We should probably have created new config files for each run, but it was easier to overwrite it and save it to Wandb instead. Especially because we changed between training locally and in the cloud.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

Training:
[Training example](figures/train_example.png)
Validation:
[Validation example](figures/validation_example.png)

As seen in the first image, we're tracking our training. We're both tracking training loss, and the accuracy as well as the AUROC score. This informs us how the training is progressing, and if we're in fact reducing loss as well as increase our accuracy over time.

The second image shows our validation samples, which are images in the validation set. We see the image, what the model predicted and what the ground truth of the image is. This is a good sanity check, as it allows us to visually confirm that the model is working.

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

We created 3 images. One for training, one for predictions/inference and one for deploying our webapp with the finished model in the cloud.

To run the web-app locally, you would run: `docker run --env-file .env -p 8501:8501 app:latest`
Here is a link to the file: <https://github.com/radziko/colo_mlops/blob/main/dockerfiles/app.dockerfile>

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

Debugging depends on who is coding. One of use used the one in PyCharm, another one of us used VSCode's debugger, and at one point we used PDB as well.
We had a memory leak, so we had to profile at one point, as the training was just slowing down and began aloccating the entire GPU memory. We found the leak :)

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:
GCP IAM: Permissions for sharing the project.
Cloud Storage: Bucket for hosting data.
Cloud Build: We use triggers here to build our docker images from PR's in Github.
Container Registry: Stores our built images from Cloud Build.
Vertex AI: Run our containers, when we're training in the cloud.
Cloud Run: Run our container for the webapp.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 50-100 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We did not use compute engine, as Vertex AI provided easier training than spinning up a whole VM, and Cloud Run is more ligthweight for hosting a simple webapp. We did not see the benefit of using Compute Engine.
### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

[Google Project buckets](figures/project_bucket.png)

[Specific data bucket](figures/colo_bucket.png)

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

[Container registry](figures/containers.png)

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We developed a webapp with Streamlit, then dockerized it, and deployed it to Cloud Run. We also tested it first locally. The webapp allows any user to upload an image of their choosing, and have the model classify their image. The top 5 probabilites are then returned.
Furthermore there are 10 example images you can classify, if you don't want to upload your own.

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

As we used the CIFAR10 dataset, there are only 10 classes. Furthermore the dataset is self-contained, and primarily used as a benchmark dataset. Therefore we don't really suspect

WE might implement it hehe

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

We used one shared GCP project, so only one member was billed in regards to the project. This member used 155 credits so far, and the most expensive service was Cloud Build, because that's where we hosted our webapp.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
> *
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 26 fill here ---

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

--- question 27 fill here ---
