#!/bin/bash
service ssh start
echo "SSH server started on port 22"
# Start the original AI-Dock service
exec /opt/ai-dock/bin/supervisor.sh