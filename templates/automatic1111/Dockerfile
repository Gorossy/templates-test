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

# Copy startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Disable ai-dock authentication and set direct access
ENV DIRECT_ADDRESS=0.0.0.0
ENV DIRECT_ADDRESS_GET_WAN=false
ENV WEB_ENABLE_AUTH=false
ENV AUTO_UPDATE=false
ENV WEBUI_PORT=27015
ENV WEBUI_FLAGS="--listen --port 27015 --api --enable-insecure-extension-access --no-gradio-queue"

EXPOSE 22 27015

CMD ["/start.sh"]