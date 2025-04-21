<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# LLM Deployment Examples using TensorRT-LLM

This directory contains examples and reference implementations for deploying Large Language Models (LLMs) in various configurations using TensorRT-LLM.


## Deployment Architectures

See [deployment architectures](../llm/README.md#deployment-architectures) to learn about the general idea of the architecture.
Note that this TensorRT-LLM version does not support all the options yet.

Note: TensorRT-LLM disaggregation does not support conditional disaggregation yet. You can only configure the deployment to always use aggregate or disaggregated serving.

## Getting Started

1. Choose a deployment architecture based on your requirements
2. Configure the components as needed
3. Deploy using the provided scripts

### Prerequisites

Start required services (etcd and NATS) using [Docker Compose](../../deploy/docker-compose.yml)
```bash
docker compose -f deploy/docker-compose.yml up -d
```

### Build docker

#### Step 1: Build TensorRT-LLM base container image

Because of the known issue of C++11 ABI compatibility within the NGC pytorch container, we rebuild TensorRT-LLM from source.
See [here](https://nvidia.github.io/TensorRT-LLM/installation/linux.html) for more informantion.

Use the helper script to build a TensorRT-LLM container base image. The script uses a specific commit id from TensorRT-LLM main branch.

```bash
# TensorRT-LLM uses git-lfs, which needs to be installed in advance.
apt-get update && apt-get -y install git git-lfs
git lfs install

./container/build_trtllm_base_image.sh
```

For more information see [here](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html#option-1-build-tensorrt-llm-in-one-step) for more details on building from source.
If you already have a TensorRT-LLM container image, you can skip this step.

#### Step 2: Build the Dynamo container

```
./container/build.sh --framework tensorrtllm
```

This build script internally points to the base container image built with step 1. If you skipped previous step because you already have the container image available, you can run the build script with that image as a base.

```bash
# Build dynamo image with other TRTLLM base image.
./container/build.sh --framework TENSORRTLLM --base-image <trtllm-base-image> --base-image-tag <trtllm-base-image-tag>
```

### Run container

```
./container/run.sh --framework tensorrtllm -it
```
## Run Deployment

This figure shows an overview of the major components to deploy:



```

+------+      +-----------+      +------------------+             +---------------+
| HTTP |----->| processor |----->|      Worker      |------------>|     Prefill   |
|      |<-----|           |<-----|                  |<------------|     Worker    |
+------+      +-----------+      +------------------+             +---------------+
                  |    ^                  |
       query best |    | return           | publish kv events
           worker |    | worker_id        v
                  |    |         +------------------+
                  |    +---------|     kv-router    |
                  +------------->|                  |
                                 +------------------+

```

Note: The above architecture illustrates all the components. The final components
that get spawned depend upon the chosen graph.

### Example architectures

#### Aggregated serving
```bash
cd /workspace/examples/tensorrt_llm
dynamo serve graphs.agg:Frontend -f ./configs/agg.yaml
```

#### Aggregated serving with KV Routing
```bash
cd /workspace/examples/tensorrt_llm
dynamo serve graphs.agg_router:Frontend -f ./configs/agg_router.yaml
```

#### Disaggregated serving
```bash
cd /workspace/examples/tensorrt_llm
TRTLLM_USE_UCX_KVCACHE=1 dynamo serve graphs.disagg:Frontend -f ./configs/disagg.yaml
```

We are defining TRTLLM_USE_UCX_KVCACHE so that TRTLLM uses UCX for transfering the KV
cache between the context and generation workers.

#### Disaggregated serving with KV Routing
```bash
cd /workspace/examples/tensorrt_llm
TRTLLM_USE_UCX_KVCACHE=1 dynamo serve graphs.disagg_router:Frontend -f ./configs/disagg_router.yaml
```

We are defining TRTLLM_USE_UCX_KVCACHE so that TRTLLM uses UCX for transfering the KV
cache between the context and generation workers.

### Client

See [client](../llm/README.md#client) section to learn how to send request to the deployment.

### Close deployment

See [close deployment](../../docs/guides/dynamo_serve.md#close-deployment) section to learn about how to close the deployment.

Remaining tasks:

- [x] Add support for the disaggregated serving.
- [ ] Add integration test coverage.
- [ ] Add instructions for benchmarking.
- [ ] Add multi-node support.
- [ ] Merge the code base with llm example to reduce the code duplication.
- [ ] Use processor from dynamo-llm framework.
- [ ] Enable NIXL integration with TensorRT-LLM once available. Currently, TensorRT-LLM uses UCX to transfer KV cache.

## KV Cache Retention Configuration

TensorRT-LLM supports retaining KV cache entries for improved performance with repeated or similar prompt sequences. This feature helps control how long KV cache entries are kept in memory and their priority for eviction.

### 1. Via YAML Configuration File

In your engine configuration YAML file, add a `kv_cache_retention_config` section:

```yaml
kv_cache_retention_config:
  priority: 50        # Numeric priority value (0-100, higher values = higher priority)
  duration_ms: 30000  # Duration in milliseconds to retain KV cache
```

### 2. Via Command Line Parameters

You can override KV cache retention settings via command line arguments:

```bash
# Set KV cache retention priority (0-100)
--local-kv-retention-priority 50

# Set KV cache retention duration in milliseconds
--local-kv-retention-duration 60000
```

### How It Works

The retention configuration works as follows:

1. The token IDs from the input request are automatically used as the keys for retention
2. The priority value (0-100) controls how likely cached entries are to be kept during memory pressure (higher values = higher priority)
3. The duration specifies how long (in milliseconds) the KV cache entries should be retained

This feature is especially useful for applications with predictable patterns of requests or where multiple users submit similar prompts.
