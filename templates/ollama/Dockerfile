FROM ollama/ollama:latest

# LLM Serving with SSH
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

# Install Python packages for LLM interaction
RUN pip3 install --no-cache-dir \
    requests \
    fastapi \
    uvicorn \
    openai \
    langchain \
    streamlit

# Create startup script
RUN echo '#!/bin/bash\nservice ssh start\nollama serve &\nsleep 5\necho "Ollama server started on port 11434"\necho "SSH server started on port 22"\necho "Main service available on port 27015"\ntail -f /dev/null' > /start.sh && chmod +x /start.sh

WORKDIR /app

EXPOSE 22 11434 27015

CMD ["/start.sh"]