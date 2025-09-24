FROM python:3.11-slim

# Oobabooga Text Generation WebUI with SSH

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

# Clone Text Generation WebUI
WORKDIR /app
RUN git clone https://github.com/oobabooga/text-generation-webui.git . && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements (check if file exists, install manually if not)
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Install additional packages
RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    bitsandbytes \
    datasets \
    peft \
    trl \
    scipy \
    gradio \
    requests \
    markdown \
    matplotlib \
    sse-starlette \
    tiktoken

# Create models directory
RUN mkdir -p models loras training/datasets training/formats characters instruction-templates

# Create startup script with optimized settings
RUN echo '#!/bin/bash\nservice ssh start\ncd /app\n# Start with optimized parameters for production (API-only mode)\npython server.py \\\n  --listen \\\n  --listen-host 0.0.0.0 \\\n  --listen-port 27015 \\\n  --api \\\n  --old-colors \\\n  --model-dir /app/models \\\n  --verbose &\necho "Oobabooga Text Generation API started with WebUI"\necho "API and WebUI available on port 27015"\necho "SSH server available on port 22"\necho "Place models in /app/models/"\ntail -f /dev/null' > /start.sh && chmod +x /start.sh

EXPOSE 22 27015

CMD ["/start.sh"]