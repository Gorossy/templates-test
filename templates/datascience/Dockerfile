FROM quay.io/jupyter/pytorch-notebook:cuda12-latest

# Data Science Stack with GPU and SSH
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

# Install additional data science packages (Python, R support, Julia, etc.)
RUN pip install --no-cache-dir \
    plotly \
    seaborn \
    scikit-learn \
    xgboost \
    lightgbm \
    dask[complete] \
    streamlit \
    fastapi \
    uvicorn \
    scipy \
    sympy \
    statsmodels \
    beautifulsoup4 \
    sqlalchemy \
    h5py

# Create startup script that works with Jupyter's entrypoint
RUN echo '#!/bin/bash\nservice ssh start' > /usr/local/bin/before-notebook.d/start-ssh.sh && \
    chmod +x /usr/local/bin/before-notebook.d/start-ssh.sh

# Set default Jupyter token (users should change this)
ENV JUPYTER_TOKEN=nodexo2025

EXPOSE 22 27015

# Use default Jupyter CMD but override port and use token from env
CMD ["start-notebook.py", "--ip=0.0.0.0", "--port=27015", "--no-browser"]