FROM python:3.10-slim

# Automatic1111 Stable Diffusion WebUI with SSH

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
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libglu1-mesa \
    libxrender1 \
    libxext6 \
    libsm6 \
    libxrandr2 \
    libfontconfig1 \
    libice6 \
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

# Clone Automatic1111 WebUI
WORKDIR /app
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git . && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    diffusers \
    xformers \
    clip \
    open-clip-torch \
    opencv-python \
    Pillow \
    requests \
    tqdm

# Create directories for models
RUN mkdir -p models/Stable-diffusion models/VAE models/ESRGAN models/GFPGAN models/RealESRGAN models/LDSR models/Lora models/ControlNet

# Create startup script
RUN echo '#!/bin/bash\nservice ssh start\necho "SSH server started on port 22"\ncd /app\necho "Starting Automatic1111 WebUI on port 27015..."\npython launch.py --listen --port 27015 --api --no-half-vae --xformers --enable-insecure-extension-access &\necho "Automatic1111 WebUI started on port 27015"\ntail -f /dev/null' > /start.sh && chmod +x /start.sh

EXPOSE 22 27015

CMD ["/start.sh"]