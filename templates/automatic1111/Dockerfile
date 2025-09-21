FROM ghcr.io/ai-dock/stable-diffusion-webui:latest

# Install SSH server
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
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

# Create startup script that starts SSH and the original service
RUN echo '#!/bin/bash\nservice ssh start\necho "SSH server started on port 22"\n# Start the original AI-Dock service\nexec /opt/ai-dock/bin/supervisor.sh' > /start.sh && chmod +x /start.sh

# Set environment variables for WebUI to listen on port 27015
ENV WEBUI_PORT=27015
ENV WEBUI_FLAGS="--listen --port 27015 --api --enable-insecure-extension-access"

EXPOSE 22 27015

CMD ["/start.sh"]