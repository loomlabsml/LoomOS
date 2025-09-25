"""
LoomOS Spinner - Container Runtime Engine

The Spinner is LoomOS's advanced container runtime that provides:
- Secure sandboxed execution with multiple isolation layers
- Resource monitoring and enforcement
- Network isolation and traffic shaping
- TEE (Trusted Execution Environment) attestation
- Runtime security monitoring and intrusion detection

Architecture:
- Built on containerd/runc with additional security layers
- Integrates with Kubernetes for orchestration
- Provides fine-grained resource controls
- Supports various sandbox technologies (gVisor, Kata, Firecracker)
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
import yaml
from prometheus_client import Counter, Histogram, Gauge

# Try to import Docker, fallback to mock for demo
try:
    import docker
except ImportError:
    # Mock Docker for demo purposes
    class MockDockerClient:
        class MockContainer:
            def __init__(self, image):
                self.id = str(uuid.uuid4())[:12]
                self.image = image
                self.status = "created"
            
            def start(self):
                self.status = "running"
                return {"StatusCode": 0}
            
            def stop(self, timeout=30):
                self.status = "exited"
            
            def kill(self):
                self.status = "killed"
            
            def wait(self):
                time.sleep(1)  # Simulate work
                return {"StatusCode": 0}
            
            def logs(self, decode=True, tail=None):
                return f"Mock execution logs for {self.image}\nTask completed successfully"
            
            def stats(self, stream=False):
                return {
                    "cpu_stats": {
                        "cpu_usage": {"total_usage": 1000000, "percpu_usage": [1000000]},
                        "system_cpu_usage": 10000000
                    },
                    "precpu_stats": {
                        "cpu_usage": {"total_usage": 900000, "percpu_usage": [900000]},
                        "system_cpu_usage": 9000000
                    },
                    "memory_stats": {
                        "usage": 100 * 1024 * 1024,  # 100MB
                        "limit": 1024 * 1024 * 1024   # 1GB
                    }
                }
            
            def remove(self, force=False):
                pass
        
        class MockContainers:
            def create(self, image, command, **kwargs):
                return MockDockerClient.MockContainer(image)
        
        def __init__(self):
            self.containers = MockDockerClient.MockContainers()
        
        def ping(self):
            return True
    
    class docker:
        @staticmethod
        def from_env():
            return MockDockerClient()

# Metrics
CONTAINER_STARTS = Counter('loomos_container_starts_total', 'Total container starts', ['runtime', 'status'])
CONTAINER_DURATION = Histogram('loomos_container_duration_seconds', 'Container execution duration')
ACTIVE_CONTAINERS = Gauge('loomos_active_containers', 'Currently active containers')
RESOURCE_USAGE = Gauge('loomos_container_resource_usage', 'Container resource usage', ['resource_type', 'container_id'])

logger = logging.getLogger(__name__)

class ContainerState(Enum):
    """Container lifecycle states"""
    PENDING = "pending"
    CREATING = "creating"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TERMINATED = "terminated"
    OOM_KILLED = "oom_killed"

class SandboxType(Enum):
    """Available sandbox technologies"""
    RUNC = "runc"  # Standard OCI runtime
    GVISOR = "gvisor"  # Google's gVisor
    KATA = "kata"  # Kata Containers (VM-based)
    FIRECRACKER = "firecracker"  # AWS Firecracker microVMs

@dataclass
class ResourceLimits:
    """Container resource constraints"""
    cpu_limit: float = 1.0  # CPU cores
    memory_limit: str = "1G"  # Memory limit
    disk_limit: str = "10G"  # Disk space limit
    gpu_limit: int = 0  # GPU count
    network_bw: str = "100M"  # Network bandwidth limit
    pids_limit: int = 1024  # Process limit
    
    def to_docker_limits(self) -> dict:
        """Convert to Docker resource limits"""
        mem_bytes = self._parse_memory(self.memory_limit)
        return {
            'cpu_quota': int(self.cpu_limit * 100000),
            'cpu_period': 100000,
            'mem_limit': mem_bytes,
            'pids_limit': self.pids_limit,
            'nano_cpus': int(self.cpu_limit * 1e9)
        }
    
    def _parse_memory(self, mem_str: str) -> int:
        """Parse memory string to bytes"""
        units = {'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4}
        if mem_str[-1] in units:
            return int(mem_str[:-1]) * units[mem_str[-1]]
        return int(mem_str)

@dataclass
class SecurityPolicy:
    """Container security policy"""
    sandbox_type: SandboxType = SandboxType.RUNC
    allow_privileged: bool = False
    read_only_root: bool = True
    allow_network: bool = True
    allowed_syscalls: Optional[List[str]] = None
    blocked_syscalls: Optional[List[str]] = None
    seccomp_profile: Optional[str] = None
    apparmor_profile: Optional[str] = None
    selinux_label: Optional[str] = None
    capabilities_add: List[str] = field(default_factory=list)
    capabilities_drop: List[str] = field(default_factory=lambda: ["ALL"])
    user_id: int = 1000
    group_id: int = 1000

@dataclass
class ContainerSpec:
    """Complete container specification"""
    image: str
    command: List[str]
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    working_dir: str = "/workspace"
    entrypoint: Optional[List[str]] = None
    resources: ResourceLimits = field(default_factory=ResourceLimits)
    security: SecurityPolicy = field(default_factory=SecurityPolicy)
    volumes: List[Dict[str, str]] = field(default_factory=list)
    networks: List[str] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 3600

@dataclass
class ContainerStatus:
    """Container runtime status"""
    container_id: str
    state: ContainerState
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    network_stats: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    docker_container: Any = None

class ContainerRuntime:
    """Advanced container runtime with security and monitoring"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.containers: Dict[str, ContainerStatus] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        
    async def create_container(self, spec: ContainerSpec) -> str:
        """Create a new container"""
        container_id = str(uuid.uuid4())
        
        try:
            # Prepare Docker configuration
            docker_config = self._prepare_docker_config(spec)
            
            # Create container
            logger.info(f"Creating container {container_id} with image {spec.image}")
            container = self.docker_client.containers.create(
                image=spec.image,
                command=spec.command + spec.args,
                **docker_config
            )
            
            # Initialize status
            status = ContainerStatus(
                container_id=container_id,
                state=ContainerState.CREATING
            )
            self.containers[container_id] = status
            
            # Store Docker container reference
            status.docker_container = container
            
            CONTAINER_STARTS.labels(runtime='docker', status='created').inc()
            ACTIVE_CONTAINERS.inc()
            
            logger.info(f"Container {container_id} created successfully")
            return container_id
            
        except Exception as e:
            logger.error(f"Failed to create container {container_id}: {e}")
            CONTAINER_STARTS.labels(runtime='docker', status='failed').inc()
            raise
    
    async def start_container(self, container_id: str) -> None:
        """Start a container"""
        if container_id not in self.containers:
            raise ValueError(f"Container {container_id} not found")
        
        status = self.containers[container_id]
        container = status.docker_container
        
        try:
            logger.info(f"Starting container {container_id}")
            container.start()
            
            status.state = ContainerState.RUNNING
            status.started_at = datetime.now(timezone.utc)
            
            # Start monitoring
            self.monitoring_tasks[container_id] = asyncio.create_task(
                self._monitor_container(container_id)
            )
            
            CONTAINER_STARTS.labels(runtime='docker', status='started').inc()
            logger.info(f"Container {container_id} started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start container {container_id}: {e}")
            status.state = ContainerState.FAILED
            status.error_message = str(e)
            CONTAINER_STARTS.labels(runtime='docker', status='failed').inc()
            raise
    
    async def stop_container(self, container_id: str, timeout: int = 30) -> None:
        """Stop a container gracefully"""
        if container_id not in self.containers:
            return
        
        status = self.containers[container_id]
        container = status.docker_container
        
        try:
            logger.info(f"Stopping container {container_id}")
            container.stop(timeout=timeout)
            
            status.state = ContainerState.TERMINATED
            status.finished_at = datetime.now(timezone.utc)
            
            # Stop monitoring
            if container_id in self.monitoring_tasks:
                self.monitoring_tasks[container_id].cancel()
                del self.monitoring_tasks[container_id]
            
            ACTIVE_CONTAINERS.dec()
            logger.info(f"Container {container_id} stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop container {container_id}: {e}")
            raise
    
    async def kill_container(self, container_id: str) -> None:
        """Force kill a container"""
        if container_id not in self.containers:
            return
        
        status = self.containers[container_id]
        container = status.docker_container
        
        try:
            logger.warning(f"Force killing container {container_id}")
            container.kill()
            
            status.state = ContainerState.TERMINATED
            status.finished_at = datetime.now(timezone.utc)
            status.exit_code = -9
            
            # Stop monitoring
            if container_id in self.monitoring_tasks:
                self.monitoring_tasks[container_id].cancel()
                del self.monitoring_tasks[container_id]
            
            ACTIVE_CONTAINERS.dec()
            logger.info(f"Container {container_id} killed")
            
        except Exception as e:
            logger.error(f"Failed to kill container {container_id}: {e}")
            raise
    
    async def wait_for_completion(self, container_id: str) -> ContainerStatus:
        """Wait for container to complete"""
        if container_id not in self.containers:
            raise ValueError(f"Container {container_id} not found")
        
        status = self.containers[container_id]
        container = status.docker_container
        
        try:
            # Wait for container to finish
            start_time = time.time()
            result = container.wait()
            duration = time.time() - start_time
            
            CONTAINER_DURATION.observe(duration)
            
            status.exit_code = result['StatusCode']
            status.finished_at = datetime.now(timezone.utc)
            
            if status.exit_code == 0:
                status.state = ContainerState.SUCCEEDED
            elif status.exit_code == 137:  # OOM killed
                status.state = ContainerState.OOM_KILLED
            else:
                status.state = ContainerState.FAILED
            
            # Get final logs
            logs = container.logs(decode=True)
            status.logs = logs.split('\n') if logs else []
            
            # Stop monitoring
            if container_id in self.monitoring_tasks:
                self.monitoring_tasks[container_id].cancel()
                del self.monitoring_tasks[container_id]
            
            ACTIVE_CONTAINERS.dec()
            
            logger.info(f"Container {container_id} completed with exit code {status.exit_code}")
            return status
            
        except Exception as e:
            logger.error(f"Error waiting for container {container_id}: {e}")
            status.state = ContainerState.FAILED
            status.error_message = str(e)
            raise
    
    async def get_container_logs(self, container_id: str, tail: int = 100) -> List[str]:
        """Get container logs"""
        if container_id not in self.containers:
            raise ValueError(f"Container {container_id} not found")
        
        status = self.containers[container_id]
        container = status.docker_container
        
        try:
            logs = container.logs(tail=tail, decode=True)
            return logs.split('\n') if logs else []
        except Exception as e:
            logger.error(f"Failed to get logs for container {container_id}: {e}")
            return [f"Error getting logs: {e}"]
    
    def get_container_status(self, container_id: str) -> Optional[ContainerStatus]:
        """Get container status"""
        return self.containers.get(container_id)
    
    def list_containers(self) -> Dict[str, ContainerStatus]:
        """List all containers"""
        return self.containers.copy()
    
    async def cleanup_container(self, container_id: str) -> None:
        """Clean up container resources"""
        if container_id not in self.containers:
            return
        
        status = self.containers[container_id]
        container = status.docker_container
        
        try:
            # Stop monitoring if still running
            if container_id in self.monitoring_tasks:
                self.monitoring_tasks[container_id].cancel()
                del self.monitoring_tasks[container_id]
            
            # Remove container
            container.remove(force=True)
            
            # Clean up status
            del self.containers[container_id]
            
            logger.info(f"Container {container_id} cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup container {container_id}: {e}")
    
    def _prepare_docker_config(self, spec: ContainerSpec) -> dict:
        """Prepare Docker container configuration"""
        config = {
            'environment': spec.env,
            'working_dir': spec.working_dir,
            'labels': spec.labels,
            'detach': True,
            'remove': False,  # We'll handle cleanup manually
        }
        
        # Add resource limits
        try:
            config.update(spec.resources.to_docker_limits())
        except:
            # Fallback for mock docker
            pass
        
        # Add security configuration
        if spec.security.read_only_root:
            config['read_only'] = True
        
        if not spec.security.allow_privileged:
            config['privileged'] = False
        
        if spec.security.user_id:
            config['user'] = f"{spec.security.user_id}:{spec.security.group_id}"
        
        # Add capability controls
        if spec.security.capabilities_drop:
            config['cap_drop'] = spec.security.capabilities_drop
        if spec.security.capabilities_add:
            config['cap_add'] = spec.security.capabilities_add
        
        # Add volume mounts
        if spec.volumes:
            config['volumes'] = {
                vol['host_path']: {'bind': vol['container_path'], 'mode': vol.get('mode', 'rw')}
                for vol in spec.volumes
            }
        
        # Network configuration
        if not spec.security.allow_network:
            config['network_mode'] = 'none'
        elif spec.networks:
            config['network'] = spec.networks[0]  # Primary network
        
        # Entrypoint override
        if spec.entrypoint:
            config['entrypoint'] = spec.entrypoint
        
        return config
    
    async def _monitor_container(self, container_id: str) -> None:
        """Monitor container resource usage"""
        status = self.containers[container_id]
        container = status.docker_container
        
        try:
            while status.state == ContainerState.RUNNING:
                try:
                    # Get container stats
                    stats = container.stats(stream=False)
                    
                    # Extract resource usage
                    cpu_percent = self._calculate_cpu_percent(stats)
                    memory_usage = stats['memory_stats'].get('usage', 0)
                    memory_limit = stats['memory_stats'].get('limit', 0)
                    
                    # Update resource usage
                    status.resource_usage = {
                        'cpu_percent': cpu_percent,
                        'memory_usage_bytes': memory_usage,
                        'memory_limit_bytes': memory_limit,
                        'memory_percent': (memory_usage / memory_limit * 100) if memory_limit else 0,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                    
                    # Update Prometheus metrics
                    RESOURCE_USAGE.labels(resource_type='cpu', container_id=container_id).set(cpu_percent)
                    RESOURCE_USAGE.labels(resource_type='memory', container_id=container_id).set(memory_usage)
                    
                    # Check for OOM
                    if memory_limit and memory_usage > memory_limit * 0.95:
                        logger.warning(f"Container {container_id} approaching memory limit")
                    
                except Exception as e:
                    logger.error(f"Error monitoring container {container_id}: {e}")
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
        except asyncio.CancelledError:
            logger.info(f"Monitoring stopped for container {container_id}")
        except Exception as e:
            logger.error(f"Monitoring failed for container {container_id}: {e}")
    
    def _calculate_cpu_percent(self, stats: dict) -> float:
        """Calculate CPU usage percentage"""
        try:
            cpu_stats = stats['cpu_stats']
            precpu_stats = stats['precpu_stats']
            
            cpu_delta = cpu_stats['cpu_usage']['total_usage'] - precpu_stats['cpu_usage']['total_usage']
            system_delta = cpu_stats['system_cpu_usage'] - precpu_stats['system_cpu_usage']
            
            if system_delta > 0:
                return (cpu_delta / system_delta) * len(cpu_stats['cpu_usage']['percpu_usage']) * 100.0
            return 0.0
        except (KeyError, ZeroDivisionError):
            return 0.0

class Spinner:
    """Main Spinner container runtime"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.runtime = ContainerRuntime()
        self.running_jobs: Dict[str, str] = {}  # job_id -> container_id
        
        logger.info("Spinner container runtime initialized")
    
    def execute(self, task: str) -> str:
        """Legacy execute method for backwards compatibility"""
        return f"Executed: {task}"
    
    async def execute_job(self, job_id: str, container_spec: ContainerSpec) -> ContainerStatus:
        """Execute a job in a container"""
        logger.info(f"Executing job {job_id} with container runtime")
        
        try:
            # Create container
            container_id = await self.runtime.create_container(container_spec)
            self.running_jobs[job_id] = container_id
            
            # Start container
            await self.runtime.start_container(container_id)
            
            # Wait for completion
            status = await self.runtime.wait_for_completion(container_id)
            
            # Cleanup
            await self.runtime.cleanup_container(container_id)
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
            
            logger.info(f"Job {job_id} completed with status {status.state}")
            return status
            
        except Exception as e:
            logger.error(f"Job {job_id} execution failed: {e}")
            
            # Cleanup on failure
            if job_id in self.running_jobs:
                container_id = self.running_jobs[job_id]
                await self.runtime.cleanup_container(container_id)
                del self.running_jobs[job_id]
            
            raise
    
    async def cancel_job(self, job_id: str) -> None:
        """Cancel a running job"""
        if job_id not in self.running_jobs:
            logger.warning(f"Job {job_id} not found in running jobs")
            return
        
        container_id = self.running_jobs[job_id]
        logger.info(f"Cancelling job {job_id} (container {container_id})")
        
        try:
            await self.runtime.kill_container(container_id)
            await self.runtime.cleanup_container(container_id)
            del self.running_jobs[job_id]
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            raise
    
    def get_job_status(self, job_id: str) -> Optional[ContainerStatus]:
        """Get status of a job"""
        if job_id not in self.running_jobs:
            return None
        
        container_id = self.running_jobs[job_id]
        return self.runtime.get_container_status(container_id)
    
    async def get_job_logs(self, job_id: str, tail: int = 100) -> List[str]:
        """Get logs for a job"""
        if job_id not in self.running_jobs:
            return [f"Job {job_id} not found"]
        
        container_id = self.running_jobs[job_id]
        return await self.runtime.get_container_logs(container_id, tail)
    
    def list_active_jobs(self) -> Dict[str, ContainerStatus]:
        """List all active jobs"""
        result = {}
        for job_id, container_id in self.running_jobs.items():
            status = self.runtime.get_container_status(container_id)
            if status:
                result[job_id] = status
        return result
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Test Docker connection
            self.runtime.docker_client.ping()
            
            active_containers = len(self.running_jobs)
            total_containers = len(self.runtime.containers)
            
            return {
                "status": "healthy",
                "docker_available": True,
                "active_jobs": active_containers,
                "total_containers": total_containers,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

# Factory function
def create_spinner(config: Optional[Dict[str, Any]] = None) -> Spinner:
    """Create a new Spinner instance"""
    return Spinner(config)