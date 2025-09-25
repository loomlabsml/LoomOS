# 🚀 LoomOS - The Iron Suit for AI Models

[![Build Status](https://github.com/loomos/loomos/workflows/CI/badge.svg)](https://github.com/loomlabsml/LoomOS/actions)
[![Coverage](https://codecov.io/gh/loomos/loomos/branch/main/graph/badge.svg)](https://codecov.io/gh/loomos/loomos)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**LoomOS** is the world's most advanced distributed AI runtime and orchestration platform. It provides enterprise-grade infrastructure for AI model deployment, training, verification, and continuous improvement at scale.

## ✨ Key Features

- 🎯 **Distributed Runtime**: Multi-node AI workload orchestration with advanced scheduling
- 🧠 **RL Training Platform**: Built-in reinforcement learning with PPO, DPO, and GRPO algorithms
- 🔍 **AI Verification Suite**: Automated safety, factuality, and quality verification
- 🔄 **Continuous Learning**: Micro-updates with LoRA/QLoRA and safe canarying
- 🌐 **Marketplace Economy**: Credit-based compute marketplace with reputation system
- 📊 **Provenance Ledger**: Immutable audit trail for all AI operations
- 🔐 **Enterprise Security**: mTLS, TEE attestation, and multi-layer sandboxing
- 🎛️ **Rich UI Dashboard**: Real-time monitoring and management interface

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- 16GB RAM minimum (32GB recommended)

### One-Command Demo
```bash
git clone https://github.com/loomos/loomos.git
cd loomos
./scripts/quickstart.sh
```

### Manual Setup
```bash
# 1. Start infrastructure
docker-compose up -d

# 2. Install dependencies
pip install -e .

# 3. Run demo workload
python examples/demo/submit_demo.py

# 4. Monitor execution
python examples/demo/monitor_job.py --job-id demo-001
```

## 📖 Documentation

- [🏗️ Architecture Guide](architecture/ARCHITECTURE.md)
- [🔧 API Reference](docs/API_REFERENCE.md)
- [🛡️ Security Guide](docs/SECURITY.md)
- [📈 Performance Tuning](docs/PERFORMANCE.md)
- [🎯 Production Deployment](docs/DEPLOYMENT.md)

## 🏢 Enterprise Features

- **Multi-Region Deployment**: Global compute distribution
- **Advanced Analytics**: Comprehensive metrics and insights
- **Custom Integrations**: REST/GraphQL APIs and webhooks
- **Professional Support**: 24/7 technical assistance
- **Compliance Tools**: SOC2, GDPR, HIPAA ready

## 📊 Performance Benchmarks

| Metric | Value |
|--------|-------|
| Job Throughput | 10,000+ jobs/hour |
| Scheduling Latency | <100ms p99 |
| Success Rate | 99.9% SLA |
| Resource Efficiency | 95%+ utilization |

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

Apache-2.0 for core platform. Enterprise features under commercial license.

## 🆘 Support

- 📧 Email: support@loomos.ai
- 💬 Discord: [Join our community](https://discord.gg/loomos)
- 📚 Docs: [docs.loomos.ai](https://docs.loomos.ai)
- 🎫 Issues: [GitHub Issues](https://github.com/loomos/loomos/issues)

## 🌍 Distributed training over the Internet

LoomOS is built to run large-scale training and evaluation jobs across machines that are not on the same private network — including cloud VMs, on-prem servers, and developer laptops. This section explains how LoomOS orchestrates multi-node training over the public internet, the connectivity and security choices, and practical examples you can use to get started.

### High-level architecture

- Control plane (Master / Nexus): a small, authoritative cluster that coordinates jobs, schedules workers, holds the job manifests and checkpoints metadata, and exposes a secure API for clients and workers.
- Data plane (Workers): compute nodes that perform training, send gradients/metrics, store local checkpoints, and stream logs and telemetry.
- Artifact storage: durable object storage (S3-compatible) used for datasets, full checkpoints, and artifacts. Keeps heavy data off the control plane.
- Optional Relay / Tunnel: when direct connectivity is blocked (NAT, strict firewalls), LoomOS supports a relay layer that brokers connections between masters and workers to traverse restrictive networks.

These components are intentionally separated so sensitive credentials and orchestration remain protected in a compact control plane while compute-heavy workloads execute on geographically distributed workers.

### Connectivity patterns

1. Cloud-hosted master with public endpoint

	- Master runs in a secure cloud VPC with a stable, TLS-protected public endpoint (load-balanced). Workers connect outbound to the master API. This is the simplest and most common deployment for internet-scale training.

2. VPN / WireGuard peering

	- Establish a private overlay between cloud and edge using WireGuard or a managed VPN. Good for steady fleets and lower latency between nodes.

3. SSH reverse tunnel or SOCKS proxy (for occasional workers)

	- Useful for developer laptops or home/edge devices: an SSH reverse tunnel lets a worker expose a port back to the master without inbound firewall changes.

4. Relay / broker service (TURN-like)

	- LoomOS provides an optional relay component for high-reliability NAT traversal. Workers and masters connect outbound to the relay, which forwards encrypted streams between parties.

Choose the pattern that matches your security posture and scale. For production, we recommend cloud-hosted master + mTLS + S3 for artifacts; for ad-hoc development, SSH tunnels or relays work well.

### Security & hardening

- mTLS for control-plane traffic: all RPCs between master, workers, and SDK clients must use mutual TLS with short-lived certificates.
- Token-based worker enrollment: workers require a one-time enrollment token issued by the control plane to join the cluster. Enrollment binds the worker identity to a certificate and policy.
- TEE & attestation: for sensitive model IP, LoomOS supports running worker code inside TEEs (Intel SGX, AMD SEV) and validating attestation before sending training data.
- Encrypted artifacts at rest: use S3 bucket encryption + KMS for checkpoint and dataset encryption.
- Least-privilege IAM: give workers only the cloud storage and telemetry permissions they need.

### Worker lifecycle (how a remote worker joins and runs jobs)

1. Provision a machine (cloud VM, bare metal, or developer laptop).
2. Install the LoomOS worker runtime (either run the `loomos/loomnode` Docker image or install the package with `pip install -e .`).
3. Obtain an API token or enrollment credential from the LoomCtl control plane (via the admin UI or control-plane admin API). LoomCtl requires Bearer tokens for protected endpoints.
4. Start the worker process using the canonical Python entrypoint `nexus/loomnode/main.py` (the same entrypoint used by the `loomos/loomnode` container). The worker registers itself with the control plane using the configured API token.
5. Once registered, the master scheduler can schedule distributed training tasks to the worker according to the job manifest constraints.

Example — start a master with Docker Compose and run a worker (exact flags implemented in `nexus/loomnode/main.py`):

```bash
# On master (cloud VM)
docker-compose up -d loomctl loomnode

# On worker (laptop or VM): install package (optional) and start the worker process
pip install -e .

# Start worker using the Python entrypoint and canonical CLI flags
python nexus/loomnode/main.py \
  --worker-id my-worker-01 \
  --rank 0 \
  --world-size 4 \
  --master-addr master.example.com \
  --master-port 29500 \
  --compression-method top_k \
  --compression-ratio 0.01

# Configure the CLI with an API token for LoomCtl (used by the CLI and by processes that read ~/.loomos/config.yaml)
loomos config set token <API_TOKEN>

# Or export the token in the environment for the process
export LOOMOS_TOKEN=<API_TOKEN>

```

Note: the control plane implements Bearer token validation (see `nexus/loomctl/app.py`) and worker registration is handled by the `/api/v1/workers/register` endpoint on the master. Token issuance is normally performed by an admin operation or identity provider — use the LoomCtl admin UI/API to mint worker enrollment tokens for production deployments.

### Submitting a distributed training job

Distributed jobs are described with a job manifest (YAML/JSON) that includes:

- number of workers and resource selectors (e.g., gpus>=1, region=us-east-1)
- networking policy and artifact locations (dataset S3 path)
- training strategy (data-parallel, model-parallel, hybrid), gradient compression settings, checkpoint intervals

Example manifest snippet (illustrative):

```yaml
job_id: my-distributed-ppo
replicas: 6
resources:
  gpu: 1
  cpu: 8
  memory: 32Gi
strategy:
  type: data-parallel
  sync: async
artifacts:
  dataset: s3://my-bucket/datasets/mini-grid/
  output: s3://my-bucket/outputs/my-distributed-ppo/
```

Submit the job using the SDK or CLI (example):

```bash
# CLI (illustrative)
loomos jobs submit --manifest job.yaml

# Python SDK (illustrative)
from loomos_sdk import LoomOSClient
client = LoomOSClient(url='https://master.example.com', token='<API_TOKEN>')
client.submit_job('job.yaml')
```

### Networking and performance considerations

- Bandwidth & latency: gradient exchange is bandwidth-sensitive. Use gradient compression, accumulate gradients locally (larger local batches), or use asynchronous strategies for high-latency links.
- Checkpoint frequently to object storage to survive transient worker disconnects.
- Use affinity hints to co-locate workers with datasets (or use region tagging in the manifest).
- Configure worker heartbeat and aggressive health checks to drain jobs quickly from flaky home/edge workers.

### Fault tolerance & recovery

- Epoch-level checkpointing: design training loops to be restartable from the latest S3 checkpoint.
- Re-sharding: when a worker is lost, the master can reassign replicas or reschedule from a warm standby pool.
- Canary and safe rollouts: deploy small canary jobs to new worker pools before moving production workloads.

### Observability & debugging

- Metrics: Prometheus metrics for per-worker GPU/CPU, network I/O, and training throughput.
- Traces: distributed tracing (Jaeger/OpenTelemetry) to surface RPC latencies across the internet.
- Logs: central log aggregation (ELK/Cloud Logging) with per-job and per-worker tags.

### Example: WireGuard-based secure overlay (recommended for small fleets)

1. Create a WireGuard server in the cloud (on the master VPC).
2. Add worker clients with short-lived keys and policies restricting S3/Control-plane access.
3. Workers connect over WireGuard and then register with the master privately.

WireGuard pros: simple, high-performance, low latency. Cons: requires key management and some network setup.

### Common pitfalls and troubleshooting

- NAT/firewall blocks outbound non-HTTP ports: use HTTPS-based tunnels or the Relay service.
- High bandwidth costs: move datasets to region-local storage or use dataset streaming/sharding.
- Worker clocks skewed: enable NTP; clock skew can break certificate validation and checkpoint timestamps.
- Long-running GPU drivers: use container images pinned to tested driver versions.

### Best practices checklist

- Use mTLS and short-lived enrollment tokens for worker security.
- Store heavy artifacts in S3 and keep the control plane lightweight.
- Prefer cloud-hosted masters with horizontal replicas behind a load balancer for production.
- Tune training strategy for high-latency workers: increase local batch sizes, enable gradient compression, or use asynchronous strategies.
- Monitor network costs and colocate large datasets with compute when possible.

---

If you'd like, I can add a small runnable example that shows: (A) starting a master in a cloud-like environment via Docker Compose, (B) starting a worker that tunnels over SSH, and (C) submitting a multi-worker job manifest — ready-made for local testing. Tell me which connectivity pattern you'd like me to demonstrate and I'll add the runnable example to the repository.
