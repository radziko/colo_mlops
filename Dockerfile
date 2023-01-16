FROM nvcr.io/nvidia/pytorch:22.08-py3

# Update and upgrade the dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

# Copy all files
COPY . .
COPY data/ data/

# Install source as package
RUN pip install -e .

# Run train
ENTRYPOINT ["python", "src/models/train_model.py"]
