#!/bin/bash

# Start SSH server in background
service ssh start &
echo "SSH server started on port 22"

# Set ai-dock environment variables to disable auth
export WEB_ENABLE_AUTH=false
export DIRECT_ADDRESS=0.0.0.0
export DIRECT_ADDRESS_GET_WAN=false
export AUTO_UPDATE=false

# Set WebUI specific variables
export WEBUI_PORT=27015
export WEBUI_FLAGS="--listen --port 27015 --api --enable-insecure-extension-access --no-gradio-queue"

echo "Starting ai-dock with Automatic1111 WebUI on port 27015..."

# Execute the original ai-dock init script with our environment
if [ -f /opt/ai-dock/bin/init.sh ]; then
    exec /opt/ai-dock/bin/init.sh
else
    echo "ai-dock init script not found, starting WebUI directly..."
    cd /opt/stable-diffusion-webui 2>/dev/null || cd /workspace/stable-diffusion-webui 2>/dev/null || cd /app
    exec python webui.py --listen --port 27015 --api --enable-insecure-extension-access --no-gradio-queue
fi