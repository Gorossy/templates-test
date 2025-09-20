FROM python:3.10-slim

# ComfyUI - Stable Diffusion Interface with SSH

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
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
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

# Clone ComfyUI
WORKDIR /app
RUN git clone https://github.com/comfyanonymous/ComfyUI.git . && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# Install additional dependencies
RUN pip install --no-cache-dir \
    xformers \
    accelerate \
    transformers \
    diffusers

# Create startup script
RUN echo '#!/bin/bash\nservice ssh start\ncd /app\npython main.py --listen 0.0.0.0 --port 27015 &\necho "ComfyUI started on port 27015"\necho "SSH server started on port 22"\ntail -f /dev/null' > /start.sh && chmod +x /start.sh

EXPOSE 22 27015

CMD ["/start.sh"]