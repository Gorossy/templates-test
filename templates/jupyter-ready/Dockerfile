FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# JupyterLab GPU Ready with SSH

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
# Ensure PATH includes Conda binaries
ENV PATH="/opt/conda/bin:$PATH"

# Install SSH server and necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    openssh-client \
    sudo \
    curl \
    wget \
    git \
    build-essential \
    nodejs \
    npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Force "python3" to be the conda Python
RUN ln -sf /opt/conda/bin/python /usr/local/bin/python3

# Configure SSH
RUN mkdir -p /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#ListenAddress 0.0.0.0/ListenAddress 0.0.0.0/' /etc/ssh/sshd_config

# SSH access files (empty authorized_keys)
RUN mkdir -p /root/.ssh/ && echo '' > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys

# Install JupyterLab and ML packages
RUN pip install --no-cache-dir \
    jupyterlab \
    jupyter \
    ipywidgets \
    matplotlib \
    seaborn \
    plotly \
    pandas \
    numpy \
    scikit-learn \
    transformers \
    accelerate \
    datasets \
    wandb \
    tensorboard \
    opencv-python \
    Pillow \
    tqdm \
    requests

# Install JupyterLab extensions (using modern approach)
RUN pip install --no-cache-dir jupyterlab-widgets

# Create startup script
RUN echo '#!/bin/bash\nservice ssh start\ncd /app\njupyter lab --ip=0.0.0.0 --port=27015 --no-browser --allow-root --NotebookApp.token="" --NotebookApp.password="" --NotebookApp.allow_origin="*" &\necho "JupyterLab started on port 27015"\necho "SSH server started on port 22"\ntail -f /dev/null' > /start.sh && chmod +x /start.sh

# Activate Conda environment on shell startup
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate base" >> /root/.bashrc

WORKDIR /app

EXPOSE 22 27015

CMD ["/start.sh"]