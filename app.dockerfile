FROM python:3.9-slim

EXPOSE 8501

# Update and upgrade the dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY requirements_app.txt requirements_app.txt
RUN pip install --upgrade pip
RUN pip install -r requirements_app.txt --no-cache-dir

# Copy src files
COPY src/ src/

# copy app
COPY app/ app/

# Install source as package
COPY setup.py setup.py
RUN pip install -e .

# Run train
ENTRYPOINT ["streamlit", "run", "app/Upload.py", "--server.port=8501", "--server.address=0.0.0.0"]
