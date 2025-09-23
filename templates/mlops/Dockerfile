FROM python:3.11-slim

# MLOps & Monitoring with SSH

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

# Install MLOps and monitoring packages (simplified for compatibility)
RUN pip install --no-cache-dir \
    mlflow \
    wandb \
    neptune-client \
    dvc \
    fastapi \
    uvicorn \
    prometheus-client \
    docker \
    streamlit \
    dash \
    plotly \
    pandas \
    numpy \
    scikit-learn \
    requests \
    psutil

# Create MLflow directory
RUN mkdir -p /mlflow

# Create startup script for MLflow
RUN echo '#!/bin/bash\nservice ssh start\nmlflow server --host 0.0.0.0 --port 27015 --backend-store-uri file:///mlflow &\necho "MLflow server started on port 27015"\necho "SSH server started on port 22"\ntail -f /dev/null' > /start.sh && chmod +x /start.sh

WORKDIR /app

EXPOSE 22 27015

CMD ["/start.sh"]