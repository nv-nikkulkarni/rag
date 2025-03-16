<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# NeMo Guardrails Setup for RAG Blueprint

This guide provides step-by-step instructions to enable **NeMo Guardrails** for the RAG Blueprint, allowing you to control and safeguard LLM interactions.

## Overview

NeMo Guardrails is a framework that provides safety and security measures for LLM applications. When enabled, it provides:
- Content safety filtering
- Topic control to prevent off-topic conversations
- Jailbreak detection to prevent prompt attacks

## Hardware Requirements

The NeMo Guardrails models have specific hardware requirements:

- **Llama 3.1 NemoGuard 8B Content Safety Model**: Requires 48 GB of GPU memory
- **Llama 3.1 NemoGuard 8B Topic Control Model**: Requires 48 GB of GPU memory

NVIDIA developed and tested these microservices using H100 and A100 GPUs.

For detailed hardware compatibility and support information:
- [Llama 3.1 NemoGuard 8B Content Safety Support Matrix](https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-contentsafety/latest/support-matrix.html)
- [Llama 3.1 NemoGuard 8B Topic Control Support Matrix](https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-topiccontrol/latest/support-matrix.html)

---

## Setting Up NeMo Guardrails

### Prerequisites
- Docker and Docker Compose installed
- NVIDIA API Key configured
- Docker configured for GPU access
- RAG Server must be running before starting NeMo Guardrails services

---

### Deployment Options

#### Option 1: Self-hosted Deployment (Default)

This is the default deployment method that runs all guardrails services locally on your hardware.

### Step 1: Enable Guardrails

Set the environment variable to enable guardrails:

```bash
export ENABLE_GUARDRAILS=true
export DEFAULT_CONFIG=nemoguard
```

After setting these environment variables, you must restart the RAG server for `ENABLE_GUARDRAILS` to take effect:

```bash
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

---

### Step 2: Create Model Cache Directory

Create a directory for caching models (if not already created):

```bash
mkdir -p ~/.cache/model-cache
```

---

### Step 3: Set Model Directory Path

Set the model directory path:

```bash
export MODEL_DIRECTORY=~/.cache/model-cache
```

---

### Step 4: Start NeMo Guardrails Service

Start the NeMo Guardrails service using Docker Compose:

```bash
USERID=$(id -u) docker compose -f deploy/compose/docker-compose-nemo-guardrails.yaml up -d
```

This command starts the following services:
- NeMo Guardrails microservice
- Content safety model
- Topic control model

**Note:** The NemoGuard services may take several minutes to fully initialize. You can monitor their status with:

```bash
watch -n 2 'docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "nemoguard|guardrails"'
```

Wait until you see all services showing as "healthy" before proceeding:

```
llama-3.1-nemoguard-8b-topic-control    Up 5 minutes (healthy)
llama-3.1-nemoguard-8b-content-safety   Up 5 minutes (healthy)
nemo-guardrails-microservice            Up 4 minutes (healthy)
```

---

### Step 5: Enable Guardrails from the UI

Once the services are running, you can enable guardrails from the RAG Playground UI:

1. Open the RAG Playground UI
2. Go to Settings by clicking on the top right corner of the UI
3. In the "Output Preferences" section, toggle "Guardrails" to ON (as shown in the screenshot below)

![Guardrails toggle in Output Preferences](./assets/toggle_nemo_guardrails.png)

---

#### Option 2: Cloud Deployment

For cloud deployment using NVIDIA-hosted models instead of the default self-hosted deployment:

```bash
# Set configuration for cloud deployment
export DEFAULT_CONFIG=nemoguard_cloud
export NIM_ENDPOINT_URL=https://integrate.api.nvidia.com/v1

# Start only the guardrails microservice
docker compose -f deploy/compose/docker-compose-nemo-guardrails.yaml up -d nemo-guardrails-microservice
```

**Note:** Before starting the cloud deployment, verify that the model names in the configuration file are correct:

```bash
cat deploy/compose/nemoguardrails/config-store/nemoguard_cloud/config.yml
```

Ensure that the model names in this file match the models available in your NVIDIA API account. You may need to update these names based on the specific models you have access to.

---

## Current Limitations

- The Jailbreak detection model is currently not available. This feature will be added in future updates.
- For cloud deployment, only the guardrails microservice is needed; content safety and topic control services are provided through NVIDIA's cloud infrastructure.

---

## Troubleshooting

### GPU Device ID Issues

If you encounter GPU device errors, you can customize the GPU device IDs used by the guardrails services by setting these environment variables before starting the service:

```bash
# Use specific GPUs for guardrail services (default is GPU 6 and 7)
export CONTENT_SAFETY_GPU_ID=0
export TOPIC_CONTROL_GPU_ID=1
```

### Service Health Check

To verify if the guardrails services are running properly:

```bash
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "guardrails|safety|topic"
```

```bash
nemo-guardrails-microservice            Up 19 minutes
llama-3.1-nemoguard-8b-topic-control    Up 19 minutes
llama-3.1-nemoguard-8b-content-safety   Up 19 minutes
```

---

## Additional Information

For more information about NeMo Guardrails, visit the [NeMo Guardrails documentation](https://docs.nvidia.com/nemo-guardrails/).

## References

- [NeMo Guardrails Microservice Overview](https://developer.nvidia.com/docs/nemo-microservices/guardrails/source/overview.html) - Detailed information about the NeMo Guardrails microservice architecture and capabilities
- [Integrating with NemoGuard NIM Microservices](https://developer.nvidia.com/docs/nemo-microservices/guardrails/source/guides/integrate-nim.html) - Guide for integrating NemoGuard NIM microservices into your application

---