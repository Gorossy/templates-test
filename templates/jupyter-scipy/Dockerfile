FROM quay.io/jupyter/pytorch-notebook:cuda12-latest

# SciPy Stack with GPU and SSH
USER root

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install SSH server
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

# Install scipy stack packages
RUN pip install --no-cache-dir \
    scipy \
    sympy \
    patsy \
    statsmodels \
    cloudpickle \
    dill \
    numba \
    bokeh

# Create startup script that works with Jupyter's entrypoint
RUN echo '#!/bin/bash\nservice ssh start' > /usr/local/bin/before-notebook.d/start-ssh.sh && \
    chmod +x /usr/local/bin/before-notebook.d/start-ssh.sh

# Set default Jupyter token (users should change this)
ENV JUPYTER_TOKEN=nodexo2025

# Expose ports
EXPOSE 22 27015

# Use default Jupyter CMD but override port and use token from env
CMD ["start-notebook.py", "--ip=0.0.0.0", "--port=27015", "--no-browser"]