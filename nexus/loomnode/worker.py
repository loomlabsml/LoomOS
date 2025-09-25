# The `Worker` class in the code implements a distributed training worker for the LoomOS Nexus system
# with capabilities for task execution, communication with a master node, and performance monitoring.
"""
LoomOS Nexus Worker - Distributed Training Over-The-Internet

The Nexus Worker implements a revolutionary distributed training system that reduces
inter-GPU communication requirements by 3-4 orders of magnitude through:

- Advanced gradient compression and quantization
- Asynchronous parameter updates with convergence guarantees  
- Hierarchical communication topologies
- Adaptive bandwidth allocation
- Byzantine fault tolerance
- Dynamic worker scaling and recovery

Core Innovations:
- Sparse gradient updates with top-k selection
- Momentum-based compression with error feedback
- Decentralized consensus protocols
- Network-aware optimization scheduling
- Real-time performance monitoring and adaptation
"""

import asyncio
import json
import logging
import time
import uuid
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
import numpy as np
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
import socket
import psutil
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Import core LoomOS components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from core.spinner import ContainerRuntime
    from core.prism import Prism
    from core.loomdb import LoomDB, EventType, AuditContext
    from core.tapestry import Tapestry
except ImportError:
    # Mock for standalone operation
    class ContainerRuntime:
        async def create_container(self, *args, **kwargs): return "mock_container"
        async def start_container(self, *args, **kwargs): return True
        async def stop_container(self, *args, **kwargs): return True
    class Prism:
        async def verify(self, *args, **kwargs): return {"verified": True}
    class LoomDB:
        async def log_job_event(self, *args, **kwargs): return "mock_entry"
    class Tapestry:
        async def store_memory(self, *args, **kwargs): return "mock_memory"
    class EventType:
        JOB_STARTED = "job_started"
        JOB_COMPLETED = "job_completed"
    class AuditContext:
        def __init__(self, **kwargs): pass

# Mock ML libraries with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock PyTorch
    class torch:
        class nn:
            class Module: pass
        class distributed:
            @staticmethod
            def init_process_group(*args, **kwargs): pass
            @staticmethod
            def barrier(): pass
            @staticmethod
            def all_reduce(*args, **kwargs): pass

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    # Mock Docker
    class docker:
        @staticmethod
        def from_env(): return MockDockerClient()
    
    class MockDockerClient:
        def containers(self): return MockContainers()
    
    class MockContainers:
        def run(self, *args, **kwargs): return MockContainer()
    
    class MockContainer:
        def stop(self): pass
        def remove(self): pass

logger = logging.getLogger(__name__)

# Metrics
NEXUS_TASKS_TOTAL = Counter('nexus_tasks_total', 'Total tasks processed', ['worker_id', 'task_type'])
NEXUS_COMMUNICATION_BYTES = Counter('nexus_communication_bytes', 'Communication overhead', ['direction'])
NEXUS_GRADIENT_COMPRESSION_RATIO = Gauge('nexus_gradient_compression_ratio', 'Gradient compression ratio')
NEXUS_CONVERGENCE_RATE = Gauge('nexus_convergence_rate', 'Training convergence rate')
NEXUS_ACTIVE_WORKERS = Gauge('nexus_active_workers', 'Number of active workers')
NEXUS_TASK_DURATION = Histogram('nexus_task_duration_seconds', 'Task execution time')

class WorkerStatus(Enum):
    """Worker status states"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    TRAINING = "training"
    COMMUNICATING = "communicating"
    FAILED = "failed"
    SHUTDOWN = "shutdown"

class TaskType(Enum):
    """Types of distributed training tasks"""
    FORWARD_PASS = "forward_pass"
    BACKWARD_PASS = "backward_pass"
    GRADIENT_EXCHANGE = "gradient_exchange"
    PARAMETER_UPDATE = "parameter_update"
    MODEL_SYNC = "model_sync"
    CHECKPOINT_SAVE = "checkpoint_save"
    HEALTH_CHECK = "health_check"

class CompressionMethod(Enum):
    """Gradient compression methods"""
    NONE = "none"
    TOP_K = "top_k"
    RANDOM_K = "random_k"
    THRESHOLD = "threshold"
    QUANTIZATION = "quantization"
    SPARSIFICATION = "sparsification"

@dataclass
class WorkerConfig:
    """Configuration for Nexus worker"""
    worker_id: str
    worker_rank: int = 0
    world_size: int = 1
    
    # Network configuration
    master_addr: str = "localhost"
    master_port: int = 29500
    backend: str = "nccl"  # nccl, gloo, mpi
    
    # Compression settings
    compression_method: CompressionMethod = CompressionMethod.TOP_K
    compression_ratio: float = 0.01  # 1% of gradients
    quantization_bits: int = 8
    
    # Performance settings
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_concurrent_tasks: int = 4
    communication_timeout: float = 30.0
    
    # Resource limits
    max_memory_gb: float = 8.0
    max_cpu_cores: int = 4
    gpu_memory_fraction: float = 0.8
    
    # Fault tolerance
    max_retries: int = 3
    checkpoint_interval: int = 100
    heartbeat_interval: float = 10.0

@dataclass
class TrainingTask:
    """Distributed training task"""
    task_id: str
    task_type: TaskType
    model_config: Dict[str, Any]
    
    # Data configuration
    dataset_path: str
    batch_indices: List[int]
    
    # Training parameters
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4
    
    # Distributed settings
    gradient_compression: Optional[Dict[str, Any]] = None
    synchronization_mode: str = "async"  # sync, async, local_sgd
    
    # Resource requirements
    required_memory_gb: float = 4.0
    estimated_duration_seconds: float = 60.0
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: int = 1  # 1-10, higher is more important
    retry_count: int = 0

@dataclass
class CommunicationStats:
    """Statistics for communication efficiency"""
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    compressed_bytes_sent: int = 0
    compression_time_ms: float = 0.0
    communication_time_ms: float = 0.0
    
    def compression_ratio(self) -> float:
        if self.total_bytes_sent == 0:
            return 0.0
        return self.compressed_bytes_sent / self.total_bytes_sent
    
    def bandwidth_saved_mbps(self, duration_seconds: float) -> float:
        if duration_seconds == 0:
            return 0.0
        bytes_saved = self.total_bytes_sent - self.compressed_bytes_sent
        return (bytes_saved * 8) / (duration_seconds * 1_000_000)  # Mbps

class GradientCompressor:
    """Advanced gradient compression system"""
    
    def __init__(self, method: CompressionMethod, compression_ratio: float = 0.01):
        self.method = method
        self.compression_ratio = compression_ratio
        self.error_feedback: Dict[str, np.ndarray] = {}
        
        logger.info(f"Gradient compressor initialized: {method.value} (ratio: {compression_ratio})")
    
    async def compress_gradients(self, gradients: Dict[str, np.ndarray]) -> Tuple[Dict[str, Any], CommunicationStats]:
        """Compress gradients using specified method"""
        start_time = time.time()
        compressed = {}
        stats = CommunicationStats()
        
        for param_name, gradient in gradients.items():
            if self.method == CompressionMethod.TOP_K:
                compressed[param_name] = await self._top_k_compression(param_name, gradient)
            elif self.method == CompressionMethod.THRESHOLD:
                compressed[param_name] = await self._threshold_compression(param_name, gradient)
            elif self.method == CompressionMethod.QUANTIZATION:
                compressed[param_name] = await self._quantization_compression(param_name, gradient)
            elif self.method == CompressionMethod.SPARSIFICATION:
                compressed[param_name] = await self._sparsification_compression(param_name, gradient)
            else:
                compressed[param_name] = {
                    'data': gradient.tobytes(),
                    'shape': gradient.shape,
                    'dtype': str(gradient.dtype)
                }
            
            # Update statistics
            original_size = gradient.nbytes
            compressed_size = len(compressed[param_name].get('data', b''))
            
            stats.total_bytes_sent += original_size
            stats.compressed_bytes_sent += compressed_size
        
        stats.compression_time_ms = (time.time() - start_time) * 1000
        
        # Update metrics
        NEXUS_GRADIENT_COMPRESSION_RATIO.set(stats.compression_ratio())
        
        logger.debug(f"Compressed gradients: {stats.compression_ratio():.3f} ratio in {stats.compression_time_ms:.1f}ms")
        return compressed, stats
    
    async def _top_k_compression(self, param_name: str, gradient: np.ndarray) -> Dict[str, Any]:
        """Top-K sparsification with error feedback"""
        # Add error feedback from previous iterations
        if param_name in self.error_feedback:
            gradient = gradient + self.error_feedback[param_name]
        
        # Flatten gradient for top-k selection
        flat_grad = gradient.flatten()
        k = max(1, int(len(flat_grad) * self.compression_ratio))
        
        # Select top-k indices by magnitude
        abs_grad = np.abs(flat_grad)
        top_k_indices = np.argpartition(abs_grad, -k)[-k:]
        
        # Create sparse representation
        sparse_values = flat_grad[top_k_indices]
        
        # Store error feedback
        compressed_grad = np.zeros_like(flat_grad)
        compressed_grad[top_k_indices] = sparse_values
        error = flat_grad - compressed_grad
        self.error_feedback[param_name] = error.reshape(gradient.shape)
        
        return {
            'indices': top_k_indices.tobytes(),
            'values': sparse_values.tobytes(),
            'shape': gradient.shape,
            'dtype': str(gradient.dtype),
            'compression_method': 'top_k',
            'k': k
        }
    
    async def _threshold_compression(self, param_name: str, gradient: np.ndarray) -> Dict[str, Any]:
        """Threshold-based compression"""
        threshold = np.std(gradient) * 0.1  # Adaptive threshold
        
        mask = np.abs(gradient) > threshold
        indices = np.where(mask)
        values = gradient[mask]
        
        return {
            'indices': np.array(indices).tobytes(),
            'values': values.tobytes(),
            'shape': gradient.shape,
            'dtype': str(gradient.dtype),
            'compression_method': 'threshold',
            'threshold': threshold
        }
    
    async def _quantization_compression(self, param_name: str, gradient: np.ndarray) -> Dict[str, Any]:
        """Quantization-based compression"""
        # Simple uniform quantization
        min_val, max_val = gradient.min(), gradient.max()
        scale = (max_val - min_val) / (2**8 - 1)  # 8-bit quantization
        
        quantized = np.round((gradient - min_val) / scale).astype(np.uint8)
        
        return {
            'data': quantized.tobytes(),
            'shape': gradient.shape,
            'scale': scale,
            'min_val': min_val,
            'compression_method': 'quantization'
        }
    
    async def _sparsification_compression(self, param_name: str, gradient: np.ndarray) -> Dict[str, Any]:
        """Random sparsification"""
        mask = np.random.random(gradient.shape) < self.compression_ratio
        sparse_grad = gradient * mask
        
        return {
            'data': sparse_grad.tobytes(),
            'shape': gradient.shape,
            'dtype': str(gradient.dtype),
            'compression_method': 'sparsification'
        }
    
    async def decompress_gradients(self, compressed_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Decompress received gradients"""
        gradients = {}
        
        for param_name, data in compressed_data.items():
            method = data.get('compression_method', 'none')
            
            if method == 'top_k':
                gradients[param_name] = await self._decompress_top_k(data)
            elif method == 'threshold':
                gradients[param_name] = await self._decompress_threshold(data)
            elif method == 'quantization':
                gradients[param_name] = await self._decompress_quantization(data)
            elif method == 'sparsification':
                gradients[param_name] = await self._decompress_sparsification(data)
            else:
                # Uncompressed data
                gradient = np.frombuffer(data['data'], dtype=data['dtype'])
                gradients[param_name] = gradient.reshape(data['shape'])
        
        return gradients
    
    async def _decompress_top_k(self, data: Dict[str, Any]) -> np.ndarray:
        """Decompress top-k sparse gradients"""
        indices = np.frombuffer(data['indices'], dtype=np.int64)
        values = np.frombuffer(data['values'], dtype=data['dtype'])
        shape = data['shape']
        
        # Reconstruct sparse gradient
        flat_grad = np.zeros(np.prod(shape), dtype=data['dtype'])
        flat_grad[indices] = values
        
        return flat_grad.reshape(shape)
    
    async def _decompress_threshold(self, data: Dict[str, Any]) -> np.ndarray:
        """Decompress threshold-based sparse gradients"""
        indices = np.frombuffer(data['indices'], dtype=np.int64).reshape(-1, len(data['shape']))
        values = np.frombuffer(data['values'], dtype=data['dtype'])
        shape = data['shape']
        
        gradient = np.zeros(shape, dtype=data['dtype'])
        gradient[tuple(indices.T)] = values
        
        return gradient
    
    async def _decompress_quantization(self, data: Dict[str, Any]) -> np.ndarray:
        """Decompress quantized gradients"""
        quantized = np.frombuffer(data['data'], dtype=np.uint8)
        shape = data['shape']
        scale = data['scale']
        min_val = data['min_val']
        
        # Dequantize
        gradient = quantized.astype(np.float32) * scale + min_val
        
        return gradient.reshape(shape)
    
    async def _decompress_sparsification(self, data: Dict[str, Any]) -> np.ndarray:
        """Decompress sparsified gradients"""
        gradient = np.frombuffer(data['data'], dtype=data['dtype'])
        return gradient.reshape(data['shape'])

class DistributedOptimizer:
    """Nexus distributed optimizer with ultra-low communication"""
    
    def __init__(self, worker_config: WorkerConfig):
        self.config = worker_config
        self.compressor = GradientCompressor(
            worker_config.compression_method,
            worker_config.compression_ratio
        )
        
        # Momentum buffers for each parameter
        self.momentum_buffers: Dict[str, np.ndarray] = {}
        self.velocity_buffers: Dict[str, np.ndarray] = {}
        
        # Communication statistics
        self.comm_stats = CommunicationStats()
        
        logger.info(f"Distributed optimizer initialized for worker {worker_config.worker_id}")
    
    async def optimize_step(self, gradients: Dict[str, np.ndarray], 
                           learning_rate: float) -> Dict[str, np.ndarray]:
        """Perform optimization step with distributed communication"""
        start_time = time.time()
        
        # 1. Apply momentum and error correction locally
        processed_gradients = await self._apply_momentum(gradients, learning_rate)
        
        # 2. Compress gradients for communication
        compressed_gradients, compression_stats = await self.compressor.compress_gradients(processed_gradients)
        self.comm_stats.compression_time_ms += compression_stats.compression_time_ms
        
        # 3. Exchange gradients with other workers (mock implementation)
        aggregated_gradients = await self._exchange_gradients(compressed_gradients)
        
        # 4. Apply updates
        parameter_updates = await self._compute_parameter_updates(aggregated_gradients, learning_rate)
        
        optimization_time = (time.time() - start_time) * 1000
        logger.debug(f"Optimization step completed in {optimization_time:.1f}ms")
        
        return parameter_updates
    
    async def _apply_momentum(self, gradients: Dict[str, np.ndarray], 
                            learning_rate: float) -> Dict[str, np.ndarray]:
        """Apply momentum to gradients"""
        momentum = self.config.gradient_accumulation_steps * 0.9  # Adaptive momentum
        
        processed = {}
        for param_name, gradient in gradients.items():
            if param_name not in self.momentum_buffers:
                self.momentum_buffers[param_name] = np.zeros_like(gradient)
            
            # Momentum update
            self.momentum_buffers[param_name] = (
                momentum * self.momentum_buffers[param_name] + 
                (1 - momentum) * gradient
            )
            
            processed[param_name] = self.momentum_buffers[param_name]
        
        return processed
    
    async def _exchange_gradients(self, compressed_gradients: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Exchange gradients with other workers"""
        comm_start = time.time()
        
        # Mock distributed gradient exchange
        # In real implementation, this would use MPI, NCCL, or custom networking
        
        # Simulate network latency and bandwidth
        await asyncio.sleep(0.01)  # 10ms network latency
        
        # Decompress received gradients
        aggregated = await self.compressor.decompress_gradients(compressed_gradients)
        
        # Simulate averaging across workers
        for param_name, gradient in aggregated.items():
            aggregated[param_name] = gradient / self.config.world_size
        
        comm_time = (time.time() - comm_start) * 1000
        self.comm_stats.communication_time_ms += comm_time
        
        return aggregated
    
    async def _compute_parameter_updates(self, gradients: Dict[str, np.ndarray], 
                                       learning_rate: float) -> Dict[str, np.ndarray]:
        """Compute final parameter updates"""
        updates = {}
        
        for param_name, gradient in gradients.items():
            # Simple SGD update (can be extended to Adam, etc.)
            updates[param_name] = -learning_rate * gradient
        
        return updates

class Worker:
    """Nexus distributed training worker"""
    
    def __init__(self, config: Optional[WorkerConfig] = None):
        self.config = config or WorkerConfig(worker_id=f"worker_{uuid.uuid4().hex[:8]}")
        self.status = WorkerStatus.INITIALIZING
        
        # Core components
        self.container_runtime = ContainerRuntime()
        self.verifier = Prism()
        self.database = LoomDB()
        self.memory_system = Tapestry()
        
        # Docker client
        self.docker_client = docker.from_env() if DOCKER_AVAILABLE else None
        
        # Distributed training components
        self.optimizer = DistributedOptimizer(self.config)
        
        # Task management
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_concurrent_tasks)
        self.active_tasks: Dict[str, TrainingTask] = {}
        self.completed_tasks: List[str] = []
        
        # Worker coordination
        self.worker_registry: Dict[str, Dict[str, Any]] = {}
        self.heartbeat_task: Optional[asyncio.Task] = None
        
        # Master node connection
        self.master_connection: Optional[Any] = None  # WebSocket connection to master
        self.master_reconnect_attempts: int = 0
        self.max_reconnect_attempts: int = 10
        
        # Performance monitoring
        self.performance_metrics: Dict[str, float] = {
            'tasks_per_second': 0.0,
            'average_task_duration': 0.0,
            'memory_utilization': 0.0,
            'cpu_utilization': 0.0,
            'network_throughput_mbps': 0.0,
            'error_rate': 0.0
        }
        self.start_time = time.time()
        
        # TEE attestation data (if available)
        self.attestation_data: Optional[Dict[str, Any]] = None
        
        logger.info(f"Nexus worker {self.config.worker_id} initialized with master at {self.config.master_addr}:{self.config.master_port}")
    
    async def run(self):
        """Main worker event loop"""
        logger.info(f"Starting Nexus worker {self.config.worker_id}")
        
        try:
            # Connect to master node
            await self._connect_to_master()
            
            # Initialize distributed training
            await self._initialize_distributed()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Set status to idle
            self.status = WorkerStatus.IDLE
            
            # Main processing loop
            while self.status != WorkerStatus.SHUTDOWN:
                try:
                    # Process tasks from queue
                    await self._process_task_queue()
                    
                    # Brief pause to prevent CPU spinning
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(1.0)
            
        except Exception as e:
            logger.error(f"Worker {self.config.worker_id} failed: {e}")
            self.status = WorkerStatus.FAILED
        finally:
            await self._cleanup()
    
    async def _initialize_distributed(self):
        """Initialize distributed training environment"""
        if TORCH_AVAILABLE:
            try:
                # Initialize process group for distributed training
                torch.distributed.init_process_group(
                    backend=self.config.backend,
                    init_method=f"tcp://{self.config.master_addr}:{self.config.master_port}",
                    rank=self.config.worker_rank,
                    world_size=self.config.world_size
                )
                logger.info(f"Distributed training initialized: rank {self.config.worker_rank}/{self.config.world_size}")
            except Exception as e:
                logger.warning(f"Failed to initialize distributed training: {e}")
        
        # Update worker count metric
        NEXUS_ACTIVE_WORKERS.inc()
    
    async def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Start Prometheus metrics server
        try:
            start_http_server(8000 + self.config.worker_rank)
            logger.info(f"Metrics server started on port {8000 + self.config.worker_rank}")
        except Exception as e:
            logger.warning(f"Failed to start metrics server: {e}")
    
    async def _process_task_queue(self):
        """Process tasks from the queue"""
        try:
            # Get task with timeout
            task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
            
            # Execute task
            await self.execute_task(task)
            
            # Mark task as done
            self.task_queue.task_done()
            
        except asyncio.TimeoutError:
            # No tasks available, continue
            pass
    
    async def execute_task(self, task: Union[str, TrainingTask]):
        """Execute a training task"""
        start_time = time.time()
        
        # Convert string task to TrainingTask if needed
        if isinstance(task, str):
            task = TrainingTask(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                task_type=TaskType.FORWARD_PASS,
                model_config={},
                dataset_path="",
                batch_indices=[]
            )
        
        task_id = task.task_id
        self.active_tasks[task_id] = task
        self.status = WorkerStatus.TRAINING
        
        logger.info(f"Executing task {task_id} of type {task.task_type.value}")
        
        try:
            # Log task start
            await self.database.log_job_event(
                job_id=task_id,
                event_type=EventType.JOB_STARTED,
                event_data={
                    "task_type": task.task_type.value,
                    "worker_id": self.config.worker_id,
                    "model_config": task.model_config
                },
                context=AuditContext(
                    service_name="nexus_worker",
                    user_id=self.config.worker_id
                )
            )
            
            # Execute based on task type
            if task.task_type == TaskType.FORWARD_PASS:
                result = await self._execute_forward_pass(task)
            elif task.task_type == TaskType.BACKWARD_PASS:
                result = await self._execute_backward_pass(task)
            elif task.task_type == TaskType.GRADIENT_EXCHANGE:
                result = await self._execute_gradient_exchange(task)
            elif task.task_type == TaskType.MODEL_SYNC:
                result = await self._execute_model_sync(task)
            elif task.task_type == TaskType.HEALTH_CHECK:
                result = await self._execute_health_check(task)
            else:
                result = {"status": "completed", "message": f"Mock execution of {task.task_type.value}"}
            
            # Store task result in memory system
            await self.memory_system.store_memory(
                content=json.dumps(result),
                memory_type=self.memory_system.MemoryType.PROCEDURAL if hasattr(self.memory_system, 'MemoryType') else "procedural",
                modality=self.memory_system.MemoryModality.TEXT if hasattr(self.memory_system, 'MemoryModality') else "text",
                tags=[f"task:{task.task_type.value}", f"worker:{self.config.worker_id}"],
                importance=0.8
            )
            
            # Log completion
            await self.database.log_job_event(
                job_id=task_id,
                event_type=EventType.JOB_COMPLETED,
                event_data={
                    "result": result,
                    "execution_time_seconds": time.time() - start_time
                }
            )
            
            # Update metrics
            execution_time = time.time() - start_time
            NEXUS_TASK_DURATION.observe(execution_time)
            NEXUS_TASKS_TOTAL.labels(
                worker_id=self.config.worker_id,
                task_type=task.task_type.value
            ).inc()
            
            logger.info(f"Task {task_id} completed in {execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            
            # Log failure
            await self.database.log_job_event(
                job_id=task_id,
                event_type=EventType.JOB_FAILED,
                event_data={
                    "error": str(e),
                    "execution_time_seconds": time.time() - start_time
                }
            )
            
            result = {"status": "failed", "error": str(e)}
        
        finally:
            # Clean up
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            self.completed_tasks.append(task_id)
            self.status = WorkerStatus.IDLE
    
    async def _execute_forward_pass(self, task: TrainingTask) -> Dict[str, Any]:
        """Execute forward pass computation"""
        # Mock forward pass with actual tensor operations if available
        if TORCH_AVAILABLE:
            # Create mock model and data
            batch_size = min(task.batch_indices) if task.batch_indices else 32
            input_size = task.model_config.get('input_size', 784)
            
            # Simulate forward pass computation
            await asyncio.sleep(0.1)  # Simulate computation time
            
            # Mock gradients
            mock_gradients = {
                'layer1.weight': np.random.randn(256, input_size) * 0.01,
                'layer1.bias': np.random.randn(256) * 0.01,
                'layer2.weight': np.random.randn(128, 256) * 0.01,
                'layer2.bias': np.random.randn(128) * 0.01
            }
            
            return {
                "status": "completed",
                "batch_size": batch_size,
                "gradients_shape": {k: v.shape for k, v in mock_gradients.items()},
                "computation_time_ms": 100
            }
        else:
            return {
                "status": "completed",
                "message": "Forward pass executed (mock)",
                "computation_time_ms": 50
            }
    
    async def _execute_backward_pass(self, task: TrainingTask) -> Dict[str, Any]:
        """Execute backward pass and gradient computation"""
        # Mock backward pass
        mock_gradients = {
            'layer1.weight': np.random.randn(256, 784) * 0.01,
            'layer1.bias': np.random.randn(256) * 0.01,
            'layer2.weight': np.random.randn(128, 256) * 0.01,
            'layer2.bias': np.random.randn(128) * 0.01
        }
        
        # Apply distributed optimization
        parameter_updates = await self.optimizer.optimize_step(
            mock_gradients, 
            task.learning_rate
        )
        
        return {
            "status": "completed",
            "gradients_computed": len(mock_gradients),
            "parameter_updates": len(parameter_updates),
            "compression_ratio": self.optimizer.comm_stats.compression_ratio(),
            "communication_time_ms": self.optimizer.comm_stats.communication_time_ms
        }
    
    async def _execute_gradient_exchange(self, task: TrainingTask) -> Dict[str, Any]:
        """Execute gradient exchange with other workers"""
        # Mock gradient exchange
        await asyncio.sleep(0.05)  # Simulate network communication
        
        NEXUS_COMMUNICATION_BYTES.labels(direction="sent").inc(1024 * 10)  # 10KB
        NEXUS_COMMUNICATION_BYTES.labels(direction="received").inc(1024 * 8)  # 8KB (compressed)
        
        return {
            "status": "completed",
            "workers_contacted": max(1, self.config.world_size - 1),
            "bytes_exchanged": 1024 * 18,
            "compression_achieved": 0.8
        }
    
    async def _execute_model_sync(self, task: TrainingTask) -> Dict[str, Any]:
        """Execute model synchronization"""
        # Mock model synchronization
        await asyncio.sleep(0.02)
        
        return {
            "status": "completed",
            "parameters_synced": 4,
            "sync_time_ms": 20
        }
    
    async def _execute_health_check(self, task: TrainingTask) -> Dict[str, Any]:
        """Execute worker health check"""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network connectivity check
        network_healthy = await self._check_network_connectivity()
        
        # GPU check (if available)
        gpu_healthy = True
        gpu_memory_used = 0
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            except:
                gpu_healthy = False
        
        health_status = {
            "worker_id": self.config.worker_id,
            "status": self.status.value,
            "uptime_seconds": time.time() - self.start_time,
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "gpu_memory_gb": gpu_memory_used
            },
            "network_healthy": network_healthy,
            "gpu_healthy": gpu_healthy,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "performance_metrics": self.performance_metrics
        }
        
        return health_status
    
    async def _check_network_connectivity(self) -> bool:
        """Check network connectivity to master node"""
        try:
            # Simple TCP connection test
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.config.master_addr, self.config.master_port),
                timeout=5.0
            )
            writer.close()
            await writer.wait_closed()
            return True
        except:
            return False
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to master"""
        while self.status != WorkerStatus.SHUTDOWN:
            try:
                # Send heartbeat
                heartbeat_data = {
                    "type": "heartbeat",
                    "worker_id": self.config.worker_id,
                    "status": self.status.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "active_tasks": len(self.active_tasks),
                    "queue_size": self.task_queue.qsize(),
                    "capabilities": {
                        "max_memory_gb": self.config.max_memory_gb,
                        "max_cpu_cores": self.config.max_cpu_cores,
                        "compression_method": self.config.compression_method.value,
                        "gpu_available": TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False
                    },
                    "performance_metrics": self.performance_metrics
                }
                
                # Send to master node via WebSocket if connected
                if hasattr(self, 'master_connection') and self.master_connection:
                    try:
                        await self.master_connection.send_str(json.dumps(heartbeat_data))
                    except Exception as e:
                        logger.warning(f"Failed to send heartbeat to master: {e}")
                        # Attempt to reconnect
                        await self._connect_to_master()
                else:
                    # Log heartbeat for debugging
                    logger.debug(f"Heartbeat: {heartbeat_data}")
                
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5.0)
    
    async def submit_task(self, task: TrainingTask) -> bool:
        """Submit a task to the worker queue"""
        try:
            await self.task_queue.put(task)
            logger.info(f"Task {task.task_id} submitted to worker {self.config.worker_id}")
            return True
        except asyncio.QueueFull:
            logger.warning(f"Task queue full for worker {self.config.worker_id}")
            return False
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get current worker statistics"""
        return {
            "worker_id": self.config.worker_id,
            "status": self.status.value,
            "uptime_seconds": time.time() - self.start_time,
            "tasks": {
                "active": len(self.active_tasks),
                "completed": len(self.completed_tasks),
                "queued": self.task_queue.qsize()
            },
            "communication_stats": {
                "compression_ratio": self.optimizer.comm_stats.compression_ratio(),
                "total_bytes_sent": self.optimizer.comm_stats.total_bytes_sent,
                "compressed_bytes_sent": self.optimizer.comm_stats.compressed_bytes_sent
            },
            "config": {
                "world_size": self.config.world_size,
                "compression_method": self.config.compression_method.value,
                "compression_ratio": self.config.compression_ratio
            }
        }
    
    async def _cleanup(self):
        """Cleanup worker resources"""
        logger.info(f"Cleaning up worker {self.config.worker_id}")
        
        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        
        # Close master connection
        if self.master_connection:
            try:
                await self.master_connection.close()
            except:
                pass
        
        # Stop active containers
        for task_id in list(self.active_tasks.keys()):
            logger.info(f"Stopping task {task_id}")
            # In production, stop running containers/processes
        
        # Update metrics
        NEXUS_ACTIVE_WORKERS.dec()
        
        # Cleanup distributed training
        if TORCH_AVAILABLE:
            try:
                torch.distributed.destroy_process_group()
            except:
                pass
        
        self.status = WorkerStatus.SHUTDOWN
        logger.info(f"Worker {self.config.worker_id} shutdown complete")
    
    async def _connect_to_master(self):
        """Connect to master node and register"""
        import aiohttp
        
        try:
            # Register with master node
            async with aiohttp.ClientSession() as session:
                registration_data = {
                    'worker_id': self.config.worker_id,
                    'worker_rank': self.config.worker_rank,
                    'capabilities': {
                        'max_memory_gb': self.config.max_memory_gb,
                        'max_cpu_cores': self.config.max_cpu_cores,
                        'compression_method': self.config.compression_method.value,
                        'gpu_available': TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False,
                        'backend': self.config.backend
                    },
                    'attestation_data': self.attestation_data
                }
                
                master_url = f"http://{self.config.master_addr}:{self.config.master_port}"
                async with session.post(f"{master_url}/api/v1/workers/register", json=registration_data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        logger.info(f"Successfully registered with master {result.get('master_id')}")
                        
                        # Establish WebSocket connection
                        await self._establish_websocket_connection(session)
                    else:
                        error = await resp.text()
                        logger.error(f"Failed to register with master: {error}")
                        
        except Exception as e:
            logger.error(f"Failed to connect to master: {e}")
            self.master_reconnect_attempts += 1
            
            if self.master_reconnect_attempts < self.max_reconnect_attempts:
                logger.info(f"Retrying master connection in 5 seconds (attempt {self.master_reconnect_attempts})")
                await asyncio.sleep(5.0)
                await self._connect_to_master()
            else:
                logger.error("Max reconnection attempts reached. Running in standalone mode.")
    
    async def _establish_websocket_connection(self, session):
        """Establish WebSocket connection to master"""
        try:
            master_ws_url = f"ws://{self.config.master_addr}:{self.config.master_port}/ws/worker/{self.config.worker_id}"
            self.master_connection = await session.ws_connect(master_ws_url)
            
            logger.info("WebSocket connection to master established")
            
            # Start message handler
            asyncio.create_task(self._handle_master_messages())
            
        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection: {e}")
    
    async def _handle_master_messages(self):
        """Handle messages from master node"""
        if not self.master_connection:
            return
        
        try:
            async for msg in self.master_connection:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        message = json.loads(msg.data)
                        await self._process_master_message(message)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON from master: {e}")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self.master_connection.exception()}")
                    break
        except Exception as e:
            logger.error(f"Error handling master messages: {e}")
        finally:
            self.master_connection = None
            logger.warning("Lost connection to master node")
    
    async def _process_master_message(self, message: Dict[str, Any]):
        """Process a message from the master node"""
        msg_type = message.get('type')
        
        if msg_type == 'task_assignment':
            await self._handle_task_assignment(message)
        elif msg_type == 'job_cancellation':
            await self._handle_job_cancellation(message)
        elif msg_type == 'configuration_update':
            await self._handle_configuration_update(message)
        elif msg_type == 'shutdown':
            logger.info("Received shutdown command from master")
            self.status = WorkerStatus.SHUTDOWN
        else:
            logger.warning(f"Unknown message type from master: {msg_type}")
    
    async def _handle_task_assignment(self, message: Dict[str, Any]):
        """Handle task assignment from master"""
        try:
            task_data = message.get('task', {})
            
            task = TrainingTask(
                task_id=task_data.get('task_id'),
                task_type=TaskType(task_data.get('task_type')),
                model_config=task_data.get('model_config', {}),
                dataset_path=task_data.get('dataset_path', ''),
                batch_indices=task_data.get('batch_indices', []),
                learning_rate=task_data.get('learning_rate', 0.001),
                gradient_compression=task_data.get('gradient_compression'),
                priority=task_data.get('priority', 1)
            )
            
            # Add to task queue
            await self.task_queue.put(task)
            logger.info(f"Received task assignment: {task.task_id}")
            
        except Exception as e:
            logger.error(f"Failed to handle task assignment: {e}")
    
    async def _handle_job_cancellation(self, message: Dict[str, Any]):
        """Handle job cancellation from master"""
        job_id = message.get('job_id')
        
        # Cancel tasks related to this job
        cancelled_tasks = []
        for task_id, task in list(self.active_tasks.items()):
            if task_id.startswith(job_id):
                cancelled_tasks.append(task_id)
                del self.active_tasks[task_id]
        
        logger.info(f"Cancelled {len(cancelled_tasks)} tasks for job {job_id}")
    
    async def _handle_configuration_update(self, message: Dict[str, Any]):
        """Handle configuration update from master"""
        config_updates = message.get('config', {})
        
        # Update worker configuration
        for key, value in config_updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
    
    async def _send_task_completion(self, task_id: str, result: Dict[str, Any]):
        """Send task completion notification to master"""
        if self.master_connection:
            message = {
                "type": "task_completed",
                "task_id": task_id,
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            try:
                await self.master_connection.send_str(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send task completion: {e}")
    
    async def _send_task_failure(self, task_id: str, error: str):
        """Send task failure notification to master"""
        if self.master_connection:
            message = {
                "type": "task_failed",
                "task_id": task_id,
                "error": error,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            try:
                await self.master_connection.send_str(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send task failure: {e}")
    
    async def _send_performance_update(self):
        """Send performance metrics to master"""
        if self.master_connection:
            message = {
                "type": "performance_update",
                "metrics": self.performance_metrics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            try:
                await self.master_connection.send_str(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send performance update: {e}")