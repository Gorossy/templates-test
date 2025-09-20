FROM tensorflow/serving:latest

# TensorFlow Serving with SSH
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
    python3 \
    python3-pip \
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

# Install TensorFlow client and related packages
RUN pip3 install --no-cache-dir \
    tensorflow \
    tensorflow-serving-api \
    requests \
    numpy \
    pandas \
    matplotlib

# Create models directory
RUN mkdir -p /models

# Create startup script
RUN echo '#!/bin/bash\nservice ssh start\ntensorflow_model_server --rest_api_port=27015 --model_config_file=/models/models.config &\necho "TensorFlow Serving started on port 27015"\necho "SSH server started on port 22"\ntail -f /dev/null' > /start.sh && chmod +x /start.sh

WORKDIR /app

EXPOSE 22 8500 27015

CMD ["/start.sh"]