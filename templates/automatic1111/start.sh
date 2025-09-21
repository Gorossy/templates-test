#!/bin/bash
service ssh start
echo "SSH server started on port 22"

# Find and execute the original entrypoint
if [ -f /opt/ai-dock/bin/init.sh ]; then
    exec /opt/ai-dock/bin/init.sh
elif [ -f /usr/local/bin/init.sh ]; then
    exec /usr/local/bin/init.sh
elif [ -f /init ]; then
    exec /init
else
    echo "Starting Automatic1111 WebUI directly..."
    cd /workspace/stable-diffusion-webui || cd /app || cd /
    exec python launch.py --listen --port 27015 --api --enable-insecure-extension-access
fi