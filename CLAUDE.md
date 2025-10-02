# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains Docker templates for VM deployments on Bittensor infrastructure. The company provides GPU-enabled VMs to users, and these templates allow customers to quickly deploy pre-configured ML/AI environments with consistent SSH access and web interfaces.

Each template in `templates/` is a self-contained Docker environment designed to run on customer VMs with SSH access and specific ML/AI tooling pre-configured.

## Architecture

### Template Structure

Each template directory contains:
- `Dockerfile` - Container definition with SSH server + application setup
- `docker-compose.yml` - Service configuration with GPU support and port mappings

All templates follow a consistent pattern:
- **SSH Port**: Port 22 must always be exposed (typically mapped to 4444:22 in docker-compose)
- **Application Port**: Port 27015 is the **ONLY** port available for web UIs/APIs - this is a hard infrastructure constraint
- **GPU Support**: NVIDIA GPU pass-through enabled by default for customer VMs
- **SSH Configuration**: Public key authentication only (no password auth)
- **Working Directory**: `/app` for application code

### Critical Port Requirements

**IMPORTANT**: The Bittensor VM infrastructure has strict port limitations:
- **Port 22** (SSH): REQUIRED - must always be available for customer access
- **Port 27015**: ONLY port available for web interfaces, APIs, or any user-facing services
- Any template with a web UI, Jupyter interface, API endpoint, or web service MUST use port 27015 internally
- No other ports are exposed to users - design all services accordingly

### Template Categories

**ML Development Environments:**
- `pytorch/` - PyTorch + CUDA runtime base (minimal, SSH-only)
- `jupyter-ready/` - JupyterLab with PyTorch, transformers, ML packages
- `jupyter-scipy/`, `jupyter-tensorflow/`, `jupyter-spark/` - Specialized Jupyter variants
- `datascience/` - Multi-language notebook environment (Python, R, Julia)
- `scientific/` - Miniconda for custom scientific computing environments

**Model Serving & LLM Inference:**
- `ollama/` - Local LLM serving with Ollama
- `tensorflow/` - TensorFlow Serving for production model deployment
- `oobabooga/` - Text Generation WebUI for LLM interaction
- `llm-webui-qwen3-8b/`, `llm-webui-qwen3-32b/` - Qwen3 model interfaces

**Image Generation:**
- `automatic1111/` - Stable Diffusion WebUI (full-featured)
- `comfyui/` - ComfyUI for visual workflow-based Stable Diffusion

**Development Tools:**
- `vscode-web/` - VS Code in browser with ML/AI extensions
- `mlops/` - MLflow and experiment tracking tools

### Template Configuration

`templates/template_config.json` defines all available templates with:
- Template name and description
- Base Docker image
- Estimated size
- Use case description
- Directory reference

### Common Dockerfile Patterns

**Base Images:**
- PyTorch templates use `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`
- Python-based apps use `python:3.10-slim` or `python:3.11-slim`
- Data science uses official Jupyter images

**SSH Setup** (consistent across all templates):
```dockerfile
# Configure SSH for key-based auth only
RUN mkdir -p /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config
```

**Startup Scripts:**
Most templates use `/start.sh` that:
1. Starts SSH service (port 22 - required for customer access)
2. Launches the main application on port 27015 (web UI, Jupyter, API server - MUST use port 27015)
3. Keeps container running with `tail -f /dev/null` or similar

**Port Configuration Examples:**
- JupyterLab: `jupyter lab --port=27015`
- ComfyUI: `python main.py --listen 0.0.0.0 --port 27015`
- Automatic1111: `python launch.py --listen --port 27015`
- Oobabooga: `python server.py --listen-port 27015`

**Conda Environments** (PyTorch-based images):
- PATH includes `/opt/conda/bin`
- Python symlinked to conda Python
- Auto-activation in `.bashrc` for interactive shells

## Development Workflow

### Building a Template

```bash
cd templates/<template-name>
docker-compose build
docker-compose up -d
```

### Testing a Template

After starting:
1. SSH should be available on port 4444 (add your public key to `/root/.ssh/authorized_keys` first)
2. Web interface/API should be available on port 27015
3. GPU access can be verified with `nvidia-smi` inside container

### Adding a New Template

1. Create new directory in `templates/`
2. Add `Dockerfile` following the SSH + application pattern
3. Add `docker-compose.yml` with GPU config and standard port mappings (4444:22, 27015:27015)
4. **CRITICAL**: Ensure any web service/UI runs on port 27015 (this is the only available port for user-facing services)
5. Update `templates/template_config.json` with template metadata
6. Test build and runtime functionality
7. Verify SSH access on port 22 and web interface (if applicable) on port 27015
8. Verify GPU availability with `nvidia-smi`

### Modifying Templates

When editing Dockerfiles:
- **NEVER change the port 27015 requirement** - this is a hard infrastructure constraint for the Bittensor VM platform
- Maintain SSH configuration consistency (port 22 must always be available)
- Keep standard port mappings in docker-compose.yml (4444:22, 27015:27015)
- Preserve GPU pass-through in docker-compose.yml
- If adding new services with web interfaces, they MUST run on port 27015 (no exceptions)
- Test both SSH access (port 22) and application functionality (port 27015)

## Key Implementation Details

### Package Version Constraints

Some templates have specific version requirements:
- **ComfyUI**: Requires `numpy<2.0`, `torch==2.1.2`, `transformers<4.45`, `xformers==0.0.23.post1`
- **Automatic1111**: Uses PyTorch with CUDA 12.1 index URL
- **Jupyter templates**: Use modern `jupyterlab-widgets` (not deprecated jupyter-labextension)

### Startup Behavior

Templates have different startup modes:
- **SSH-only** (pytorch): Just runs sshd daemon
- **Background + SSH** (jupyter-ready, comfyui, oobabooga): Start service in background, then keep container alive
- **Foreground script** (automatic1111): Custom start.sh runs both services

### Working Directory Convention

All application code resides in `/app` - this is where users should mount volumes or work when SSH'd in.
