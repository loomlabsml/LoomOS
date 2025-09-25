"""
LoomOS Python SDK

Enterprise-grade Python SDK for LoomOS - The Iron Suit for AI Models

This SDK provides comprehensive access to LoomOS functionality including:
- Job submission and management
- Real-time monitoring and log streaming
- Cluster management and worker coordination
- AI agent execution and model adaptation
- Marketplace integration
- WebSocket support for real-time updates

Example usage:

    from loomos_sdk import LoomClient, JobSpec

    # Create client
    client = LoomClient("http://localhost:8000", token="your-token")

    # Submit a training job
    job_spec = (JobSpec("training", "BERT Fine-tuning", "pytorch:latest")
                .command(["python", "train.py"])
                .distributed(world_size=4)
                .resources("gpu", 4))

    with client:
        job_id = client.submit_job(job_spec)
        job = client.wait_for_job(job_id)
        print(f"Job completed with status: {job['status']}")

For async usage:

    from loomos_sdk import AsyncLoomClient

    async with AsyncLoomClient("http://localhost:8000", token="your-token") as client:
        job_id = await client.submit_job(job_spec)
        async for log_entry in client.stream_logs(job_id):
            print(log_entry["message"])
"""

from .client import (
    LoomClient,
    AsyncLoomClient,
    JobSpec,
    LoomOSError,
    JobNotFoundError,
    AuthenticationError,
    APIError,
    create_training_job,
    create_inference_job,
    create_evaluation_job
)

from .job_manifest import JobManifest

__version__ = "1.0.0"
__author__ = "LoomOS Team"
__email__ = "support@loomos.ai"
__description__ = "Enterprise Python SDK for LoomOS - The Iron Suit for AI Models"

__all__ = [
    # Core client classes
    "LoomClient",
    "AsyncLoomClient",
    
    # Job specification
    "JobSpec",
    "JobManifest",
    
    # Exceptions
    "LoomOSError",
    "JobNotFoundError", 
    "AuthenticationError",
    "APIError",
    
    # Convenience functions
    "create_training_job",
    "create_inference_job",
    "create_evaluation_job",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__description__"
]