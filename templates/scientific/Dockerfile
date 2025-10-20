FROM continuumio/miniconda3:latest

# Scientific Computing with SSH

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

# Install scientific Python packages
RUN conda install -y \
    numpy \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    jupyter \
    scikit-learn \
    sympy \
    numba \
    && conda clean -ya

# Install additional packages via pip
RUN pip install --no-cache-dir \
    plotly \
    bokeh \
    dash \
    streamlit \
    fastapi \
    uvicorn

# Activate Conda environment on shell startup
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate base" >> /root/.bashrc

WORKDIR /app

EXPOSE 22 27015

# Ensure /app directory has write permissions
RUN chmod 777 /app

# Create startup script with Jupyter
RUN echo '#!/bin/bash\n\
service ssh start\n\
echo "SSH server started on port 22"\n\
echo "Starting Jupyter Lab on port 27015..."\n\
jupyter lab --ip=0.0.0.0 --port=27015 --no-browser --allow-root --NotebookApp.token="" --NotebookApp.password="" &\n\
echo "Scientific computing environment ready"\n\
echo "Jupyter Lab available at http://localhost:27015"\n\
tail -f /dev/null' > /start.sh && chmod +x /start.sh

# Default command: startup script
CMD ["/start.sh"]