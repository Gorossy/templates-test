FROM jupyter/datascience-notebook:latest

# Data Science Stack with SSH
USER root

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install SSH server and necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    openssh-client \
    sudo \
    curl \
    wget \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure SSH
RUN mkdir -p /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#ListenAddress 0.0.0.0/ListenAddress 0.0.0.0/' /etc/ssh/sshd_config

# SSH access files (empty authorized_keys)
RUN mkdir -p /root/.ssh/ && echo '' > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys

# Install additional data science packages
RUN pip install --no-cache-dir \
    plotly \
    seaborn \
    scikit-learn \
    xgboost \
    lightgbm \
    dask[complete] \
    streamlit \
    fastapi \
    uvicorn

WORKDIR /app

EXPOSE 22 27015 8888

# Start SSH daemon and Jupyter
CMD service ssh start && jupyter lab --ip=0.0.0.0 --port=27015 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''