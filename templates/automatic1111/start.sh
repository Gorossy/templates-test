#!/bin/bash

# Start SSH server in background
service ssh start
echo "SSH server started on port 22"

# Set environment variables for Stable Diffusion WebUI
export WEBUI_FLAGS="--listen --port 27015 --api --enable-insecure-extension-access --no-half-vae"
export AUTO_LAUNCH_BROWSER="False"
export CLI_ARGS="--listen --port 27015 --api --enable-insecure-extension-access --no-half-vae"

echo "Starting Automatic1111 WebUI on port 27015..."

# Try to find the webui directory and launch
if [ -d "/opt/stable-diffusion-webui" ]; then
    cd /opt/stable-diffusion-webui
    python webui.py $CLI_ARGS
elif [ -d "/workspace/stable-diffusion-webui" ]; then
    cd /workspace/stable-diffusion-webui
    python webui.py $CLI_ARGS
elif [ -d "/app/stable-diffusion-webui" ]; then
    cd /app/stable-diffusion-webui
    python webui.py $CLI_ARGS
else
    echo "Could not find stable-diffusion-webui installation"
    tail -f /dev/null
fi