FROM codercom/code-server:latest

# VS Code Web with GPU ML/AI Stack and SSH
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
    python3 \
    python3-pip \
    python3-dev \
    nodejs \
    npm \
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

# Install PyTorch first
RUN pip3 install --no-cache-dir --break-system-packages \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other ML packages
RUN pip3 install --no-cache-dir --break-system-packages \
    transformers \
    accelerate \
    datasets \
    diffusers \
    opencv-python \
    matplotlib \
    seaborn \
    pandas \
    numpy \
    scikit-learn \
    jupyter \
    jupyterlab \
    streamlit \
    gradio \
    fastapi \
    uvicorn \
    requests \
    tqdm

# Install VS Code extensions
RUN code-server --install-extension ms-python.python \
    --install-extension ms-toolsai.jupyter \
    --install-extension ms-vscode.cpptools \
    --install-extension GitHub.copilot \
    --install-extension ms-python.pylint \
    --install-extension ms-python.black-formatter

# Create startup script
RUN echo '#!/bin/bash\nservice ssh start\ncd /app\ncode-server --bind-addr 0.0.0.0:27015 --auth none --disable-telemetry &\necho "VS Code Server started on port 27015"\necho "SSH server started on port 22"\ntail -f /dev/null' > /start.sh && chmod +x /start.sh

WORKDIR /app

EXPOSE 22 27015

CMD ["/start.sh"]