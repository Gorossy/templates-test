# Template Testing Guide

This document outlines the testing procedures for each Docker template. Each template should have a separate Jira ticket with the corresponding test cases.

## General Test Cases (Apply to ALL Templates)

### 1. Basic Container Operations
- [ ] Container builds successfully
- [ ] Container starts without errors
- [ ] Container runs in detached mode
- [ ] Container restarts after failure
- [ ] Container logs are accessible

### 2. SSH Access
- [ ] SSH service starts on port 22
- [ ] SSH accepts public key authentication
- [ ] SSH denies password authentication
- [ ] Root login works with authorized key
- [ ] SSH connection is stable (no disconnects)

### 3. GPU Access (GPU-enabled templates)
- [ ] `nvidia-smi` command works inside container
- [ ] GPU is detected and accessible
- [ ] CUDA version matches expected version
- [ ] GPU memory is correctly reported
- [ ] Multiple GPUs detected if available

### 4. Port Mappings
- [ ] Port 22 (SSH) is accessible from host (mapped to 4444)
- [ ] Port 27015 (application) is accessible from host (if applicable)
- [ ] No port conflicts with other containers

### 5. Resource Management
- [ ] Container uses expected amount of disk space
- [ ] Memory usage is within acceptable limits
- [ ] CPU usage is normal during idle
- [ ] GPU memory allocation works correctly

---

## Template-Specific Test Cases

### Template 1: `nirepo/default-pytorch:2.8.0-cuda12.8-cudnn9-runtime`
**Digest:** `sha256:75c5f262ff46a0b4eb84d914c9f0740d5fc79478a366085fbfd6bce51f48cfe9`

**Description:** PyTorch base template with CUDA support and SSH access

#### Specific Tests:
- [ ] PyTorch is installed and importable
- [ ] PyTorch version is 2.8.0
- [ ] CUDA 12.8 is available
- [ ] cuDNN 9 is properly configured
- [ ] Conda environment is activated
- [ ] Python3 points to conda Python
- [ ] Can create simple PyTorch tensor on GPU
- [ ] Can run basic neural network forward pass
- [ ] Conda package manager works

#### GPU-Specific Tests:
- [ ] `torch.cuda.is_available()` returns True
- [ ] `torch.cuda.device_count()` returns correct GPU count
- [ ] Can allocate tensors on GPU
- [ ] GPU computation works (simple matrix multiplication)
- [ ] Memory transfer between CPU and GPU works

#### Test Script:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"GPU computation successful: {z.shape}")
```

---

### Template 2: `nirepo/ollama-ssh:latest`
**Digest:** `sha256:b689dee49d9b03753d6f76e5ab2f4d6b655d1080234bcc070b1f54e937f233c5`

**Description:** Ollama LLM serving with SSH access

#### Specific Tests:
- [ ] Ollama service starts successfully
- [ ] Ollama API is accessible on port 27015
- [ ] Can list available models via API
- [ ] Can pull a small test model (e.g., tinyllama)
- [ ] Can run inference on pulled model
- [ ] API responds with correct JSON format
- [ ] Ollama logs are accessible

#### GPU-Specific Tests:
- [ ] Ollama detects GPU
- [ ] Model loads on GPU (not CPU)
- [ ] GPU memory increases when model is loaded
- [ ] Inference uses GPU acceleration
- [ ] Multiple concurrent requests work on GPU

#### Test Commands:
```bash
# Check Ollama is running
curl http://localhost:27015/api/tags

# Pull a test model
curl http://localhost:27015/api/pull -d '{"name": "tinyllama"}'

# Run inference
curl http://localhost:27015/api/generate -d '{
  "model": "tinyllama",
  "prompt": "Why is the sky blue?"
}'
```

---

### Template 3: `nirepo/comfyui-ssh:latest`
**Digest:** `sha256:312411098589ca6e8b557589f4dcf5be2a2367045f724ffe807befc4bedc14f1`

**Description:** ComfyUI for Stable Diffusion workflows with SSH

#### Specific Tests:
- [ ] ComfyUI web interface loads on port 27015
- [ ] Can access ComfyUI workflow editor
- [ ] PyTorch and dependencies are installed
- [ ] Can load default workflow
- [ ] Interface is responsive
- [ ] WebSocket connection works

#### GPU-Specific Tests:
- [ ] ComfyUI detects GPU in settings
- [ ] Can load a Stable Diffusion checkpoint (if available)
- [ ] Can run image generation workflow
- [ ] GPU memory is utilized during generation
- [ ] VRAM usage is reported correctly
- [ ] xformers acceleration works

#### Test Steps:
1. Navigate to http://[host]:27015
2. Load default workflow
3. Check GPU is selected in settings
4. (If model available) Run simple txt2img generation
5. Verify generation completes without errors

---

### Template 4: `nirepo/automatic1111-ssh:latest`
**Digest:** `sha256:5a6da72aabccd04a897706052191db61876dfdc5461b51feebfb1b61912b5e6a`

**Description:** Stable Diffusion WebUI (Automatic1111) with SSH

#### Specific Tests:
- [ ] WebUI starts on port 27015
- [ ] Web interface is accessible
- [ ] Can access settings page
- [ ] API is enabled and accessible
- [ ] Model directories are created
- [ ] Extensions system works

#### GPU-Specific Tests:
- [ ] GPU is detected in system info
- [ ] CUDA version displayed correctly
- [ ] xformers is available
- [ ] Can select GPU in settings
- [ ] (If model available) Can load checkpoint on GPU
- [ ] (If model available) Can generate images using GPU
- [ ] VRAM monitoring works

#### Test Commands:
```bash
# Check WebUI is running
curl http://localhost:27015

# Check API
curl http://localhost:27015/sdapi/v1/sd-models

# Check system info
curl http://localhost:27015/sdapi/v1/memory
```

---

### Template 5: `nirepo/scientific-ssh:latest`
**Digest:** `sha256:40c93cdcba24212a71f9348b65d79ed59f113996eed36711c5cb8d240fbebc7e`

**Description:** Miniconda with scientific Python stack and SSH

#### Specific Tests:
- [ ] Miniconda is installed
- [ ] Conda commands work
- [ ] Can create new conda environments
- [ ] Base environment is activated
- [ ] Python 3 is available
- [ ] Can install packages via conda
- [ ] Can install packages via pip

#### GPU-Specific Tests:
- [ ] Can install GPU-accelerated packages (cupy, rapids, etc.)
- [ ] CUDA toolkit can be installed via conda
- [ ] GPU libraries can detect NVIDIA hardware
- [ ] Can run GPU-accelerated numpy operations (if cupy installed)

#### Test Script:
```bash
conda --version
python --version
conda list
conda create -n test_env python=3.10 -y
conda activate test_env
conda install numpy scipy matplotlib -y
python -c "import numpy; print(numpy.__version__)"
```

---

### Template 6: `nirepo/jupyter-scipy-ssh:latest`
**Digest:** `sha256:4831abb222b3ac6cdfa17b8d9a6efad10f5555cb5497548b201143e1218a923c`

**Description:** JupyterLab with SciPy stack and SSH

#### Specific Tests:
- [ ] JupyterLab starts on port 27015
- [ ] Can access JupyterLab interface
- [ ] No token/password required (as configured)
- [ ] Can create new notebook
- [ ] Python kernel works
- [ ] SciPy stack packages are available (numpy, scipy, matplotlib, pandas)
- [ ] Can execute code cells
- [ ] Can upload/download files

#### GPU-Specific Tests:
- [ ] Can import GPU libraries (if installed)
- [ ] Can run CUDA code from notebook
- [ ] Can visualize GPU memory usage
- [ ] Matplotlib plots render correctly
- [ ] Can use GPU-accelerated pandas/cuDF (if installed)

#### Test Notebook:
```python
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd

print(f"NumPy: {np.__version__}")
print(f"SciPy: {scipy.__version__}")
print(f"Pandas: {pd.__version__}")

# Test plotting
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.show()
```

---

### Template 7: `nirepo/jupyter-spark-ssh:latest`
**Digest:** `sha256:0ea465a32d093670374ffd883a39b45d2835043a0fc5507ad68e4c3838b4242f`

**Description:** JupyterLab with Apache Spark and SSH

#### Specific Tests:
- [ ] JupyterLab starts on port 27015
- [ ] Spark is installed
- [ ] Can create SparkSession
- [ ] Spark UI is accessible
- [ ] Can run basic Spark operations
- [ ] PySpark kernel works
- [ ] Java/Scala dependencies are available

#### GPU-Specific Tests:
- [ ] Spark can use GPU resources (RAPIDS Accelerator if configured)
- [ ] GPU is visible to Spark executors
- [ ] Can configure Spark to use GPU memory
- [ ] GPU-accelerated operations work (if RAPIDS installed)

#### Test Notebook:
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Test") \
    .getOrCreate()

print(f"Spark version: {spark.version}")

# Create test DataFrame
data = [("Alice", 1), ("Bob", 2)]
df = spark.createDataFrame(data, ["name", "value"])
df.show()

spark.stop()
```

---

### Template 8: `nirepo/jupyter-tensorflow-ssh:latest`
**Digest:** `sha256:a11b46549c4dd26e06c8dd733d5ecfa1384211e6e0964c38420f5021fe5a9f6e`

**Description:** JupyterLab with TensorFlow and SSH

#### Specific Tests:
- [ ] JupyterLab starts on port 27015
- [ ] TensorFlow is installed
- [ ] Can import tensorflow
- [ ] TensorFlow version is correct
- [ ] Keras is available
- [ ] Can create simple model
- [ ] Can run training

#### GPU-Specific Tests:
- [ ] `tf.config.list_physical_devices('GPU')` shows GPUs
- [ ] TensorFlow can allocate GPU memory
- [ ] Can run model training on GPU
- [ ] GPU memory growth is configurable
- [ ] Mixed precision training works
- [ ] Multi-GPU strategy works (if multiple GPUs)

#### Test Notebook:
```python
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")

# Test GPU computation
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print(f"GPU computation result:\n{c}")

# Simple model test
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
print("Model created successfully")
```

---

## Acceptance Criteria for All Templates

### Functional Requirements
- [ ] All general test cases pass
- [ ] All template-specific test cases pass
- [ ] No critical errors in logs
- [ ] Performance is acceptable
- [ ] Documentation is accurate

### Non-Functional Requirements
- [ ] Container starts in < 30 seconds (excluding model downloads)
- [ ] SSH connection established in < 5 seconds
- [ ] Web interfaces load in < 10 seconds
- [ ] GPU detection happens in < 5 seconds
- [ ] Memory leaks are not present (24-hour stability test)

### Security Requirements
- [ ] No password authentication enabled
- [ ] Only public key SSH access works
- [ ] No unnecessary ports exposed
- [ ] Container runs with appropriate permissions
- [ ] No sensitive data in logs

---

## GPU Test Environment Requirements

### Minimum Requirements
- NVIDIA GPU with CUDA Compute Capability 7.0+
- NVIDIA Driver version 525+
- nvidia-docker2 or NVIDIA Container Toolkit installed
- At least 8GB GPU memory (16GB+ recommended)

### Recommended Test GPUs
- RTX 4090 / RTX 5090
- RTX A6000
- L40 / L40s
- A100 (40GB or 80GB)
- H100

### Test Environment Setup
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Verify GPU is accessible
nvidia-smi

# Check driver version
cat /proc/driver/nvidia/version
```

---

## Reporting Template for Jira Tickets

### Ticket Structure

**Title:** `[TESTING] Template: [template-name]`

**Description:**
```
Template Image: [image:digest]
Template Type: [GPU/CPU]
Test Environment: [GPU model, driver version, CUDA version]

Test Results:
- General Tests: [X/Y passed]
- Specific Tests: [X/Y passed]
- GPU Tests: [X/Y passed]

Issues Found:
1. [Description]
2. [Description]

Performance Metrics:
- Startup time: [X seconds]
- SSH connection time: [X seconds]
- GPU detection time: [X seconds]

Recommendation: [PASS/FAIL/NEEDS_REVIEW]
```

**Labels:** `testing`, `docker`, `gpu`, `template`

**Priority:** Based on template importance

---

## Automated Testing Script Template

```bash
#!/bin/bash
# Template Testing Script

IMAGE_NAME="$1"
CONTAINER_NAME="test-${IMAGE_NAME//\//-}"

echo "Testing: $IMAGE_NAME"

# Start container
docker run -d --name "$CONTAINER_NAME" \
  --gpus all \
  -p 4444:22 \
  -p 27015:27015 \
  "$IMAGE_NAME"

# Wait for container to be ready
sleep 10

# Test SSH
echo "Testing SSH..."
timeout 5 bash -c "cat < /dev/null > /dev/tcp/localhost/4444" && echo "✓ SSH port open" || echo "✗ SSH port failed"

# Test GPU (if applicable)
echo "Testing GPU access..."
docker exec "$CONTAINER_NAME" nvidia-smi && echo "✓ GPU detected" || echo "✗ GPU not detected"

# Test application port (if applicable)
echo "Testing application port..."
timeout 5 bash -c "cat < /dev/null > /dev/tcp/localhost/27015" && echo "✓ App port open" || echo "✗ App port failed"

# Check logs for errors
echo "Checking logs for errors..."
docker logs "$CONTAINER_NAME" 2>&1 | grep -i error && echo "✗ Errors found in logs" || echo "✓ No errors in logs"

# Cleanup
docker stop "$CONTAINER_NAME"
docker rm "$CONTAINER_NAME"

echo "Testing complete"
```

---

## Notes

- All tests should be performed on clean container instances
- GPU tests require physical GPU hardware
- Some tests may require model downloads (expect longer initial run times)
- Document any deviations from expected behavior
- Include screenshots for UI-based templates
- Record performance metrics for comparison
