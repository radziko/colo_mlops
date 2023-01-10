FROM python:3.9-slim
ARG gdrive_client_id
ARG gdrive_client_secret

WORKDIR /usr/src/app

COPY . .

RUN pip install "dvc[gdrive]"
RUN dvc remote modify storage gdrive_client_id ${gdrive_client_id}
RUN dvc remote modify storage gdrive_client_secret ${gdrive_client_secret}
RUN dvc pull


FROM nvcr.io/nvidia/pytorch:22.08-py3

# Update and upgrade the dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

# Copy data files
COPY --from=0 /usr/src/app/data data/

# Copy all files
COPY . .

# Install source as package
RUN pip install -e .

# Run train
ENTRYPOINT ["python", "src/models/train_model.py"]
